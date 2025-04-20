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

import functools
import gc
import itertools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import tree
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
from oat.utils.ops import entropy_from_logits, masked_mean, masked_sum, masked_whiten

"""PPO (https://arxiv.org/abs/1707.06347) with optional KL regularization."""


@dataclass
class PPOArgs(OATArgs):
    num_ppo_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs to train."},
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
    non_stop_fixed_reward: Optional[float] = field(
        default=None,
        metadata={"help": "Fixed reward for responses not containing eos."},
    )
    reinforce_update: bool = field(
        default=False,
        metadata={"help": "The simplest REINFORCE updates."},
    )
    ignore_no_eos: bool = field(
        default=False,
        metadata={"help": "Ignore responses that cannot finish within budget."},
    )
    whiten_adv: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the advantages."},
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

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[TrajectoryData]:
        assert not self.eval_mode
        info = {}
        logging.info(f"actor start")

        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)

        candidates = []
        prompt_token_ids = []
        no_eos = []
        response_ids = []
        response_logprobs = []
        resp_lens = []
        for i in range(len(outputs)):
            # for each prompt
            prompt_token_ids.append(outputs[i].prompt_token_ids)
            candidates.append([])
            response_logprobs.append([])
            response_ids.append([])
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)
                no_eos.append(outputs[i].outputs[k].finish_reason == "length")
                token_ids = outputs[i].outputs[k].token_ids
                logps = outputs[i].outputs[k].logprobs
                logps = [item[token_ids[i]].logprob for i, item in enumerate(logps)]

                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                resp_lens.append(len(token_ids))

        info["actor/generate_time"] = time.time() - st

        # step 2. verify
        st = time.time()
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

        info["actor/verify_time"] = time.time() - st

        logging.info(f"actor reward {rewards.mean()}")
        info["actor/rewards"] = rewards.mean()
        info["actor/no_eos_count"] = no_eos.sum()
        info["actor/num_data"] = rewards.numel()
        info["actor/response_tok_len"] = np.mean(resp_lens)
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = self.sampling_params.temperature

        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                reward = rewards[i][j].item()
                if self.args.non_stop_fixed_reward is not None and no_eos[i][j]:
                    reward = self.args.non_stop_fixed_reward
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
                        loss_mask=not no_eos[i][j] if self.args.ignore_no_eos else True,
                        info=info,
                    )
                )
        logging.info(f"actor finished data_len={len(trajectory_data)}")
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle


class PPOLearner(RLLearner):
    def _init(self, args: PPOArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.args = args
        self.dataset_builder = TrajectoryDataset
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )

    def learn(self, learning_round: int):
        torch.cuda.synchronize()
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
            shuffle=(
                True if self.args.critic_type == "ppo" else False
            ),  # Do not shuffle for group MC methods (GRPO / Dr. GRPO).
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
        if self.critic is not None:
            self.critic.train()
        st = time.time()

        logging.info(
            f"start learn() buffer_len={len(self.pi_buffer)} dl_len={len(dataloader)}"
        )
        for data in dataloader:
            if local_sgd_steps > self.args.max_sgd_steps:
                break
            infos = self.learning_step(data)
            self.policy_sgd_step += (
                len(dataset)
                * self.args.num_ppo_epochs
                / self.args.train_batch_size_per_device
                / self.strategy.grad_acc_step
            )
            learn_batch_time.append(time.time() - st)
            step_bar.update()

            self.global_step += 1
            if self.global_step % self.strategy.grad_acc_step == 0:
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
        logging.info(f"finish learn()")

        return train_info

    def compute_ppo_advantages(self, rewards, input_ids, att_mask, response_masks):
        all_values = []

        with torch.no_grad():
            for i in range(0, len(input_ids), self.args.train_batch_size_per_device):
                batch_inds = torch.arange(i, i + self.args.train_batch_size_per_device)
                ## Forward critic network.
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
                all_values.append(batch_values)
        values = torch.cat(all_values)

        # Compute gae (for policy learning) and return (for critic learning); vectorize later.
        advantages = torch.zeros_like(rewards)
        for i in range(len(advantages)):
            action_inds = torch.where(response_masks[i])[0]
            lastgaelam = 0
            for t in reversed(action_inds):
                nextvalues = values[i, t + 1] if t < action_inds[-1] else 0.0
                delta = rewards[i, t] + self.args.gamma * nextvalues - values[i, t]
                lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
                advantages[i, t] = lastgaelam

        returns = advantages + values
        if self.args.whiten_adv:
            advantages = masked_whiten(advantages, response_masks)
        return advantages, returns, values

    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        rewards = rewards.sum(-1)
        # Compute monte carlo trajectory-level advantage
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if self.args.critic_type == "grpo":
            # Additionally normalize by std.
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages

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
        # action_logprobs = [
        #     torch.tensor(lp).to(device) for lp in trajectory["action_logprobs"]
        # ]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]

        logging.info(f"learn data size {input_ids.shape}")

        indices = torch.arange(
            response_masks.size(1), device=response_masks.device
        ).expand_as(response_masks)
        masked_indices = torch.where(
            response_masks, indices, torch.full_like(indices, -1)
        )
        eos_indices = masked_indices.max(dim=1).values

        # Forward old models.
        ## 1) (Option 1) Policy log probabilities are directly from actors (vLLM).
        # logps = torch.zeros_like(response_masks).float()
        # for i in range(len(logps)):
        #     logps[i, torch.where(response_masks[i])[0]] = action_logprobs[i]
        ## 2) (Option 2) Reevaluate log probabilities using learner model.
        logps = torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, device=input_ids.device
        )
        with torch.no_grad():
            for i in range(0, len(input_ids), args.train_batch_size_per_device):
                mini_batch_inds = torch.arange(i, i + args.train_batch_size_per_device)
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]

                # Remove unnecessary padding introduced by the large PPO batch.
                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]

                batch_logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ].float()
                batch_logits /= args.temperature
                batch_logps = self.get_batch_logps(
                    batch_logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                logps[mini_batch_inds, : mb_last_valid_token_pos - 1] = batch_logps

        ## 2) Reference.
        if self.ref_model is not None:
            all_ref_logps = []
            with torch.no_grad():
                for i in range(0, len(input_ids), args.train_batch_size_per_device):
                    batch_inds = torch.arange(i, i + args.train_batch_size_per_device)

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
            ref_logps = torch.cat(all_ref_logps)

            # Combine final reward and kl penalty as rewards.
            kl_rewards = -args.kl_penalty_coef * (logps - ref_logps) * response_masks
            rewards = kl_rewards.clone()
            del all_ref_logps
            torch.cuda.empty_cache()
            gc.collect()
        else:
            rewards = torch.zeros_like(response_masks).float()

        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()

        if self.args.critic_type == "ppo":
            advantages, returns, values = self.compute_ppo_advantages(
                rewards, input_ids, att_mask, response_masks
            )
        elif self.args.critic_type in ["grpo", "drgrpo"]:
            advantages = self.compute_monte_carlo_advantages(rewards, response_masks)[
                :, None
            ]

        # Compute losses and update models for multiple PPO epochs.
        stats = defaultdict(list)
        local_grad_step = 0
        for _ in range(args.num_ppo_epochs):
            batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(input_ids), args.train_batch_size_per_device):
                local_grad_step += 1
                mini_batch_inds = batch_inds[
                    b_st : b_st + args.train_batch_size_per_device
                ]
                mb_advantage = advantages[mini_batch_inds]
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_loss_masks = loss_masks[mini_batch_inds]

                # Remove unnecessary padding introduced by the large PPO batch.
                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                # # Further reduce valid token num to speed up IF:
                # ## 1. We only have PG loss, i.e., args.beta == 0.
                # ## 2. Advantage is zero in bandit case (e.g., GRPO).
                # ## 3. train_batch_size_per_device is 1.
                # if (
                #     args.beta == 0
                #     and self.args.critic_type == "grpo"
                #     and len(mb_advantage) == 1
                # ):
                #     zero_adv = (mb_advantage == 0).item()  # bool
                #     if zero_adv:
                #         mb_last_valid_token_pos = 7  # An unimportant magic number.
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]
                mb_logps = mb_logps[:, : mb_last_valid_token_pos - 1]

                if self.args.critic_type == "ppo":
                    mb_return = returns[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_values = values[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_advantage = mb_advantage[:, : mb_last_valid_token_pos - 1]

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
                if args.reinforce_update:
                    pg_loss_max = -mb_advantage * new_logps
                else:
                    logprobs_diff = new_logps - mb_logps
                    ratio = torch.exp(logprobs_diff)
                    pg_losses = -mb_advantage * ratio
                    pg_losses2 = -mb_advantage * torch.clamp(
                        ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                    )
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    stats["logprobs_diff_max"].append(
                        torch.amax(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["logprobs_diff_min"].append(
                        torch.amin(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["zero_pg_loss_count"].append(
                        (pg_loss_max == 0).detach().sum().item()
                    )

                pg_loss = self.masked_aggregator(pg_loss_max, mb_response_masks, axis=1)
                pg_loss = (pg_loss * mb_loss_masks).mean()
                infos["pg_loss"] = pg_loss.detach()
                loss = pg_loss
                if args.beta > 0:
                    mb_ref_logps = ref_logps[mini_batch_inds]
                    mb_ref_logps = mb_ref_logps[:, : mb_last_valid_token_pos - 1]
                    # k3 kl: http://joschu.net/blog/kl-approx.html.
                    # clamp to avoid numerical instability.
                    log_ratio = (mb_ref_logps - new_logps).clamp(-40.0, 40.0)
                    kl3 = torch.expm1(log_ratio) - log_ratio  # expm1 is more stable.
                    infos["kl3"] = (kl3 * mb_response_masks).detach().sum(1).mean()

                    reg_loss = self.masked_aggregator(kl3, mb_response_masks, axis=1)
                    reg_loss = args.beta * (reg_loss * mb_loss_masks).mean()
                    infos["reg_loss"] = reg_loss.detach()
                    loss += reg_loss

                entropy = entropy_from_logits(logits[:, :-1])
                entropy = masked_mean(entropy, mb_response_masks)
                infos["entropy"] = entropy.detach()

                self.strategy.backward(loss, self.model, self.optimizer)

                if local_grad_step % self.strategy.grad_acc_step == 0:
                    _st = time.time()
                    stats["policy_grad_norm"].append(
                        self.strategy.get_gradient_norm(self.model)
                    )
                    stats["get_grad_norm_time"].append(time.time() - _st)

                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                if self.args.critic_type == "ppo":
                    # torch.cuda.empty_cache()
                    # gc.collect()

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

                    vf_loss = 0.5 * self.masked_aggregator(
                        vf_loss_max, mb_response_masks, axis=1
                    )
                    critic_loss = args.vf_coef * (vf_loss * mb_loss_masks).mean()

                    self.strategy.backward(
                        critic_loss, self.critic, self.critic_optimizer
                    )
                    self.strategy.optimizer_step(
                        self.critic_optimizer, self.critic, self.critic_scheduler
                    )
                    infos["critic_loss"] = critic_loss.detach()
                    infos["vf_clipfrac"] = masked_mean(
                        (vf_losses2 > vf_losses1).float(), mb_response_masks
                    ).detach()

                with torch.no_grad():
                    if not args.reinforce_update:
                        pg_clipfrac = masked_mean(
                            (pg_losses2 > pg_losses).float(), mb_response_masks, axis=1
                        )
                        stats["pg_clipfrac"].append(pg_clipfrac.mean().min().item())

        infos.update(
            {f"{k}_nan": torch.tensor(stats[k]).isnan().sum() for k in stats.keys()}
        )
        infos.update(
            {f"{k}_inf": torch.tensor(stats[k]).isinf().sum() for k in stats.keys()}
        )
        infos["policy_grad_norm"] = torch.tensor(stats["policy_grad_norm"]).max()
        infos["get_grad_norm_time"] = torch.tensor(sum(stats["get_grad_norm_time"]))
        if not args.reinforce_update:
            infos["logprobs_diff_max"] = torch.tensor(stats["logprobs_diff_max"]).max()
            infos["logprobs_diff_min"] = torch.tensor(stats["logprobs_diff_min"]).min()
            infos["zero_pg_loss_count"] = (
                torch.tensor(stats["zero_pg_loss_count"]).float().mean()
            )
            infos["pg_clipfrac"] = torch.tensor(stats["pg_clipfrac"]).mean()
        infos["adv_mean"] = advantages.mean().cpu()
        infos["adv_min"] = advantages.min().cpu()
        infos["adv_max"] = advantages.max().cpu()
        infos["all_zero_rewards_count"] = (
            (final_rewards.view(-1, self.args.num_samples).mean(-1) == 0).sum().cpu()
        )
        infos["all_one_rewards_count"] = (
            (final_rewards.view(-1, self.args.num_samples).mean(-1) == 1).sum().cpu()
        )

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
    ):
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
