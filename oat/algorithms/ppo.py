# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal Policy Optimization."""

import gc
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import tree
import vllm
from torch.utils.data import DataLoader
from tqdm import tqdm

from oat.actors import RewardActor
from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.learners import OfflineLearner, RLLearner
from oat.types import TrajectoryData
from oat.utils.data import (
    TrajectoryDataset,
    get_datasets,
    load_data_from_disk_or_hf,
    shard_buffer,
)
from oat.utils.ops import masked_mean, masked_whiten

"""PPO (https://arxiv.org/abs/1707.06347) with additional KL regularization."""


@dataclass
class PPOArgs(OATArgs):
    num_ppo_epochs: int = field(
        default=2,
        metadata={"help": "Number of epochs to train."},
    )
    mini_train_batch_size_per_device: int = field(
        default=1,
        metadata={"help": "Mini batch size."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards."},
    )
    kl_penalty_coef: float = field(
        default=0,
        metadata={"help": "KL coefficient for pseudo rewards."},
    )
    non_stop_penalty: float = field(
        default=0,
        metadata={"help": "Penalty for responses not containing eos."},
    )
    reward_scale: float = field(
        default=1.0,
        metadata={"help": "Scaling the environment rewards."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    vf_coef: float = field(
        default=1.0,
        metadata={"help": "Value function coefficient."},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Clip range for the value function."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor."},
    )
    lam: float = field(
        default=1.0,
        metadata={"help": "Lambda value for GAE."},
    )


class PPOActor(RewardActor):
    def __init__(self, ipc_server, vllm_args, args: PPOArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            stop=["\n\nQuestion", "\n\nProblem"],
            n=args.num_samples,
            logprobs=2,
        )
        self.eval_sampling_params = vllm.SamplingParams(
            n=args.eval_n,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
            stop=["\n\nQuestion", "\n\nProblem"],
        )

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[TrajectoryData]:
        assert not self.eval_mode
        info = {}

        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)

        candidates = []
        prompt_token_ids = []
        no_eos = []
        response_ids = []
        response_logprobs = []
        for i in range(len(outputs)):
            # for each prompt
            prompt_token_ids.append(outputs[i].prompt_token_ids)
            candidates.append([])
            response_logprobs.append([])
            response_ids.append([])
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)
                no_eos.append(
                    self.tokenizer.eos_token_id not in outputs[i].outputs[k].token_ids
                )
                token_ids = outputs[i].outputs[k].token_ids
                logps = outputs[i].outputs[k].logprobs
                logps = [item[token_ids[i]].logprob for i, item in enumerate(logps)]

                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                # print(outputs[i].outputs[k].text)
                if no_eos[-1]:
                    print(outputs[i].outputs[k].token_ids)
                    print(outputs[i].outputs[k].text)

        info["actor/generate_time"] = time.time() - st

        # step 2. verify
        rewards, _ = self.oracle.get_reward(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in prompts
                )
            ),
            tree.flatten(candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in references
                )
            ),
        )
        rewards = rewards.reshape(len(prompts), -1)
        no_eos = np.array(no_eos).reshape(len(prompts), -1)

        info["actor/minibatch_accuracy"] = rewards.mean()
        info["actor/no_eos_count"] = no_eos.sum()
        info["actor/num_data"] = rewards.numel()

        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                reward = rewards[i][j]
                reward += self.args.non_stop_penalty if no_eos[i][j] else 0
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                trajectory_data.append(
                    TrajectoryData(
                        prompt=prompt,
                        prompt_ids=prompt_token_ids[i],
                        response=candidates_per_prompt[j],
                        response_ids=response_ids[i][j],
                        response_logprobs=response_logprobs[i][j],
                        rewards=dense_rewards,
                        info=info,
                    )
                )
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle


class PPOLearner(RLLearner):
    def _init(self, args: OATArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.dataset_builder = TrajectoryDataset

    def learn(self, learning_round: int):
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.strategy,
        )
        if learning_round == 1:
            self.strategy.print("Training example")
            self.strategy.print(dataset[0])

        # Load all buffered data, and PPO will iterate through inner loops.
        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        step_bar = tqdm(
            range(len(dataloader)),
            desc="Train steps",
            disable=not self.strategy.is_rank_0(),
        )
        learn_batch_time = []

        self.model.train()
        self.critic.train()
        st = time.time()
        for data in dataloader:
            if local_sgd_steps > self.args.max_sgd_steps:
                break
            infos = self.learning_step(data)
            self.policy_sgd_step += (
                len(dataset)
                * self.args.num_ppo_epochs
                / self.args.train_batch_size_per_device
                / self.strategy.accumulated_gradient
            )
            learn_batch_time.append(time.time() - st)
            step_bar.update()

            self.global_step += 1
            if self.global_step % self.strategy.accumulated_gradient == 0:

                self.gradient_update_elapse = time.time() - self.gradient_update_st
                st = time.time()
                self.gradient_update_st = time.time()

                local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        return train_info

    def learning_step(self, trajectory):
        args: PPOArgs = self.args
        infos = {}
        device = torch.cuda.current_device()
        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        final_rewards = (
            torch.tensor([r[-1] for r in trajectory["rewards"]])
            .to(device)
            .reshape(-1, 1)
        ).float() * args.reward_scale
        prompt_id_lens = trajectory["prompt_ids_lens"]
        action_logprobs = [
            torch.tensor(lp).to(device) for lp in trajectory["action_logprobs"]
        ]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]

        self.strategy.print(f"learn data size {input_ids.shape}")
        # Forward old models.
        all_ref_logps = []
        all_values = []
        with torch.no_grad():
            for i in range(0, len(input_ids), args.mini_train_batch_size_per_device):
                batch_inds = torch.arange(i, i + args.mini_train_batch_size_per_device)
                ## 1) Policy log probabilities are directly from actors.
                ## 2) Critic.
                batch_values = self.critic(
                    input_ids=input_ids[batch_inds], attention_mask=att_mask[batch_inds]
                )
                batch_value_masks = att_mask[batch_inds].clone()[:, 1:]
                batch_value_masks = torch.concat(
                    [
                        batch_value_masks,
                        torch.zeros(len(batch_value_masks), 1, device=att_mask.device),
                    ],
                    axis=1,
                )
                batch_values = (batch_values * batch_value_masks)[:, :-1]
                ## 3) Reference.
                batch_ref_logits = self.ref_model(
                    input_ids[batch_inds], attention_mask=att_mask[batch_inds]
                )["logits"].float()
                batch_ref_logits /= args.temperature
                batch_ref_logps = self.get_batch_logps(
                    batch_ref_logits,
                    input_ids[batch_inds],
                    response_masks[batch_inds],
                )

                all_ref_logps.append(batch_ref_logps)
                all_values.append(batch_values)

        ref_logps = torch.cat(all_ref_logps)
        values = torch.cat(all_values)

        logps = torch.zeros_like(ref_logps)
        for i in range(len(logps)):
            logps[i, torch.where(response_masks[i])[0]] = action_logprobs[i]

        del (all_ref_logps, all_values)
        torch.cuda.empty_cache()
        gc.collect()

        indices = torch.arange(
            response_masks.size(1), device=response_masks.device
        ).expand_as(response_masks)
        masked_indices = torch.where(
            response_masks, indices, torch.full_like(indices, -1)
        )
        eos_indices = masked_indices.max(dim=1).values

        # Combine final reward and kl penalty as rewards.
        kl_rewards = -args.kl_penalty_coef * (logps - ref_logps) * response_masks
        rewards = kl_rewards.clone()
        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()

        # Compute gae (for policy learning) and return (for critic learning); vectorize later.
        advantages = torch.zeros_like(logps)
        for i in range(len(advantages)):
            action_inds = torch.where(response_masks[i])[0]
            lastgaelam = 0
            for t in reversed(action_inds):
                nextvalues = values[i, t + 1] if t < action_inds[-1] else 0.0
                delta = rewards[i, t] + args.gamma * nextvalues - values[i, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advantages[i, t] = lastgaelam

        returns = advantages + values
        advantages = masked_whiten(advantages, response_masks)

        # Compute losses and update models for multiple PPO epochs.
        stats = defaultdict(list)
        for _ in range(args.num_ppo_epochs):
            batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(input_ids), args.mini_train_batch_size_per_device):
                mini_batch_inds = batch_inds[
                    b_st : b_st + args.mini_train_batch_size_per_device
                ]
                mb_advantage = advantages[mini_batch_inds]
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_ref_logps = ref_logps[mini_batch_inds]
                mb_return = returns[mini_batch_inds]
                mb_values = values[mini_batch_inds]
                mb_loss_masks = loss_masks[mini_batch_inds]

                # Policy learning.
                logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ].float()
                logits /= args.temperature
                new_logps = self.get_batch_logps(
                    logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                logprobs_diff = new_logps - mb_logps
                ratio = torch.exp(logprobs_diff)
                pg_losses = -mb_advantage * ratio
                pg_losses2 = -mb_advantage * torch.clamp(
                    ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                )
                pg_loss_max = torch.max(pg_losses, pg_losses2)

                stats["ratio_max"].append(ratio.detach().max().item())
                stats["ratio_min"].append(ratio.detach().min().item())

                pg_loss = masked_mean(pg_loss_max, mb_response_masks, axis=1)
                pg_loss = (pg_loss * mb_loss_masks).mean()
                loss = pg_loss
                if args.beta > 0:
                    # k3 kl: http://joschu.net/blog/kl-approx.html.
                    log_ratio = mb_ref_logps - new_logps
                    kl = torch.exp(log_ratio) - log_ratio - 1
                    kl = torch.clamp(
                        kl * mb_response_masks,
                        min=0,
                        max=10,
                    )
                    reg_loss = args.beta * kl.sum(dim=1)
                    reg_loss = (reg_loss * mb_loss_masks).mean()
                    loss += reg_loss
                    infos["reg_loss"] = reg_loss.detach()

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                infos["pg_loss"] = pg_loss.detach()

                # Critic learning.
                value_pred = self.critic(
                    input_ids=mb_input_ids, attention_mask=mb_att_mask
                )[:, :-1]

                value_pred_clipped = torch.clamp(
                    value_pred,
                    mb_values - args.cliprange_value,
                    mb_values + args.cliprange_value,
                )
                vf_losses1 = torch.square(value_pred - mb_return)
                vf_losses2 = torch.square(value_pred_clipped - mb_return)
                vf_loss_max = torch.max(vf_losses1, vf_losses2)
                vf_loss = 0.5 * masked_mean(vf_loss_max, mb_response_masks, axis=1)
                critic_loss = args.vf_coef * (vf_loss * mb_loss_masks).mean()

                self.strategy.backward(critic_loss, self.critic, self.critic_optimizer)
                self.strategy.optimizer_step(
                    self.critic_optimizer, self.critic, self.critic_scheduler
                )
                infos["critic_loss"] = critic_loss.detach()
                infos["vf_clipfrac"] = masked_mean(
                    (vf_losses2 > vf_losses1).float(), mb_response_masks
                ).detach()

        infos.update(
            {f"{k}_nan": torch.tensor(stats[k]).isnan().sum() for k in stats.keys()}
        )
        infos.update(
            {f"{k}_inf": torch.tensor(stats[k]).isinf().sum() for k in stats.keys()}
        )
        infos["ratio_max"] = torch.tensor(stats["ratio_max"]).max()
        infos["ratio_min"] = torch.tensor(stats["ratio_min"]).min()

        return infos

    def get_completion_mask(
        self,
        attention_mask: torch.LongTensor,
        prompt_id_lens: List[int],
    ):
        completion_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(completion_masks, prompt_id_lens):
            mask[:source_len] = False
        completion_masks = completion_masks
        return completion_masks

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        completion_masks: torch.LongTensor,
    ) -> torch.Tuple[torch.Tensor]:
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        # dummy token; we'll ignore the losses on these tokens later
        labels[completion_masks == False] = 0

        all_logp = logits.log_softmax(-1)
        target_logps = torch.gather(all_logp, dim=2, index=labels.unsqueeze(2)).squeeze(
            2
        )

        return target_logps


class OfflinePPOLearner(OfflineLearner, PPOLearner):
    def prepare_data(self, strategy, tokenizer):
        """Construct offline RL dataset."""
        args: PPOArgs = self.args
        data = load_data_from_disk_or_hf(args.prompt_data)[args.train_split]
        all_shards = []
        for item in tqdm(data, desc="loading data", disable=not strategy.is_rank_0()):
            all_shards.append(
                TrajectoryData(
                    prompt=item[args.input_key],
                    responses=[item[args.output_key]],  # accept a list
                    rewards=[[item[args.reward_key]]],  # accept a list
                    info={},
                )
            )

        self.all_buffer: List[TrajectoryData] = shard_buffer(
            all_shards,
            dist.get_rank(),
            dist.get_world_size(),
            args.seed,
            shuffle=True,
            drop_last=True,
        )
        self.prompts_dataset = tree.flatten(
            all_shards
        )  # needed to calculate lr scheduler
        self.prompts_dataloader = None
        if args.eval_steps > 0:
            _, self.eval_prompts_dataset = get_datasets(
                tokenizer, strategy, eval_only=True
            )
            self.eval_prompts_dataloader = DataLoader(
                self.eval_prompts_dataset,
                batch_size=strategy.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
