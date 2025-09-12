# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal Policy Optimization for multi-turn interactions.

The main difference here is that the MDP is not on token-level but
on turn-level, which changes the state/action spaces hence the target
of critic modeling.
"""

import abc
import gc
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import tree

from oat.actors.base import ActorBase
from oat.algorithms.ppo import ActorBase, PPOActor, PPOArgs, PPOLearner
from oat.collectors.base import FeedbackCollector
from oat.types import Transition, TransitionData
from oat.utils.data import (
    TransitionDataset,
)
from oat.utils.deepspeed import DeepspeedStrategy
from oat.utils.ipc import PlasmaShmClient
from oat.utils.ops import masked_mean


class MultiTurnTransitionDataset(TransitionDataset):
    def __init__(
        self,
        buffer: List[TransitionData],
        tokenizer: Callable,
        strategy: DeepspeedStrategy,
        **_,
    ) -> None:
        super().__init__(buffer, tokenizer, strategy, **_)

        # Add advantage and return calculated for multi-turn interactions.
        for i in range(len(buffer)):
            self.transitions[i]["advantages"] = buffer[i].info["advantages"]
            self.transitions[i]["returns"] = buffer[i].info["returns"]
            self.transitions[i]["values"] = buffer[i].info["values"]

    def collate_fn(self, item_list):
        batch_trajectories = super().collate_fn(item_list)
        batch_trajectories.update(
            {
                "advantages": [],
                "returns": [],
                "values": [],
            }
        )
        for t in item_list:
            batch_trajectories["advantages"].append(t["advantages"])
            batch_trajectories["returns"].append(t["returns"])
            batch_trajectories["values"].append(t["values"])
        return batch_trajectories


class MultiTurnFeedbackCollector(FeedbackCollector):
    def get_metrics(
        self,
        actor_time: float,
        feedback_data: Sequence[Sequence[Transition]],
    ):
        metric = {
            "actor/total_time": actor_time,
            "actor/num_transitions": len(tree.flatten(feedback_data)),
        }
        metric.update(
            {
                "actor/generate_avg_str_len": np.mean(
                    [len(t.response) for episode in feedback_data for t in episode]
                ),
                "actor/response_tok_len": np.mean(
                    [len(t.response_ids) for episode in feedback_data for t in episode]
                ),
                "actor/avg_reward": np.mean(
                    [t.rewards for episode in feedback_data for t in episode]
                ),
            }
        )
        mean_info = tree.map_structure(
            lambda *x: np.mean(x),
            *[t.info for episode in feedback_data for t in episode],
        )
        metric.update(mean_info)

        return metric

    def collect_feedback(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        refs: List[str],
        same_actor_group: torch.distributed.ProcessGroup,
    ):
        # generate response & get feedback
        st_time = time.time()

        rank = torch.distributed.get_rank()
        actor = self.actors[(rank // self.args.num_gpus_per_actor) % len(self.actors)]
        logging.info(
            f"Learner {rank} local actor rank: {(rank // self.args.num_gpus_per_actor) % len(self.actors)}"
        )
        assert torch.distributed.get_rank(same_actor_group) == 0
        logging.info(f"rank {rank} invoking step on actor {actor}")
        handle = actor.step(prompts, formatted_prompts, refs)
        feedback_data: Sequence[Sequence[Transition]] = self.ipc_client.deserialize_ipc(
            handle
        )

        actor_time = time.time() - st_time
        return feedback_data, self.get_metrics(actor_time, feedback_data)


class PPOMultiTurnActor(PPOActor):
    """
    We assume there is a multi-turn environment for
    interaction rather than a static prompt dataset."""

    @abc.abstractmethod
    def collect_experience(
        self,
    ) -> Tuple[Sequence[Sequence[Transition]], Dict[str, Any]]:
        """Collect experience by interacting with the environment."""

    def step(self, prompts=None, formatted_prompts=None, references=None):
        # The provided parameters are ignored since we generate prompts from the environment
        del prompts, formatted_prompts, references
        info = {}
        finished_episodes, collection_info = self.collect_experience()

        # logging infos
        info["actor/mean_episode_len"] = np.mean([len(ep) for ep in finished_episodes])
        info["actor/mean_episode_return"] = np.mean(
            [
                sum(transition.rewards for transition in episode)
                for episode in finished_episodes
            ]
        )
        info["actor/mean_episode_success"] = np.mean(
            [episode[-1].rewards == 1 for episode in finished_episodes]
        )  # NOTE: assuming success rewards is always 1

        # update collection info
        info.update(
            {k.replace("actor/", "actor/"): v for k, v in collection_info.items()}
        )
        for episode in finished_episodes:
            for transition in episode:
                transition.info.update(**info)

        # Serialize and return the trajectories
        handle = self.ipc_client.serialize_ipc(finished_episodes)
        return handle  # type: ignore


class PPOMultiTurnLearner(PPOLearner):

    def _init(self, args: PPOArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.dataset_builder = MultiTurnTransitionDataset
        self.collector = MultiTurnFeedbackCollector(
            args, actors, PlasmaShmClient(self.ipc_server)
        )
        assert (
            args.train_batch_size_per_device == 1
        ), "Please set train_batch_size_per_device = 1"
        assert args.num_gpus_per_actor == 1, "Please set num_gpus_per_actor = 1"

    def compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        gamma: float,
        lam: float,
    ):
        """
        Compute advantages and returns using either:
        - GAE if self.critic is available
        - REINFORCE if self.critic==None (ignores critic, lam=1.0)

        Args:
            rewards: [T] list/array of rewards
            values: [T+1] list/array of values (bootstrap last)
            dones: [T] list/array of done flags (0 or 1)
            gamma: discount factor
            lam: GAE lambda (ignored if use_critic=False)

        Returns:
            advantages: [T]
            returns: [T]
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        if self.critic is None:
            # For REINFORCE: values = 0, lam = 1
            values = np.zeros(T + 1, dtype=np.float32)
            lam = 1.0

        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def process_feedback_data(self, data_list: Sequence[Sequence[Transition]]):
        """
        data_list: A list of episodes, where each episode contains many transitions.

        In this function we compute important quantities for learning, e.g., advantages,
        then flatten the episodes into transition-level training samples.
        """
        flatten_transitions = []
        state_input_ids = [
            [torch.tensor(list(t.prompt_ids)) for t in episode] for episode in data_list
        ]
        state_att_masks = [
            [torch.ones(len(ids)) for ids in eids] for eids in state_input_ids
        ]

        for i, episode in enumerate(data_list):
            rewards = [t.rewards for t in episode]
            dones = [float(t.done) for t in episode]
            values = []
            if self.critic is not None:
                with torch.no_grad():
                    for inds, masks in zip(state_input_ids[i], state_att_masks[i]):
                        # Forward critic network
                        single_values = (
                            self.critic(
                                input_ids=inds.to(torch.cuda.current_device())[None],
                                attention_mask=masks.to(torch.cuda.current_device())[
                                    None
                                ],
                            )[0, 0]
                            .cpu()
                            .item()
                        )  # take the first tokens prediction
                        values.append(single_values)
            values.append(1e9)  # termination state value, set a large value to debug
            rewards, values, dones = list(map(np.array, [rewards, values, dones]))
            advantages, returns = self.compute_advantages(
                rewards, values, dones, self.args.gamma, self.args.lam
            )
            for j in range(len(episode)):
                episode[j].info.update(
                    {
                        "advantages": advantages[j],
                        "returns": returns[j],
                        "values": values[j] if self.critic is not None else 0.0,
                    }
                )
                flatten_transitions.append(episode[j])

        # Subsample trajectories if they exceed the batch size
        if len(flatten_transitions) > self.args.rollout_batch_size_per_device:
            subsample_indices = np.random.choice(
                len(flatten_transitions),
                self.args.rollout_batch_size_per_device,
                replace=False,
            )
            flatten_transitions = [flatten_transitions[si] for si in subsample_indices]

        logging.info("adding data into buffer")
        # Add to buffer
        self.pi_buffer.extend(flatten_transitions)

        # Also add to all_buffer if we're tracking all data
        if self.args.dump_all_buffer:
            self.all_buffer.extend(flatten_transitions)

        # Update query step (for tracking progress)
        self.query_step += len(flatten_transitions)

    def learning_step(self, trajectory):
        args: PPOArgs = self.args
        infos = {}
        device = torch.cuda.current_device()
        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        state_ids = tree.map_structure(lambda x: x.to(device), trajectory["state_ids"])
        final_rewards = (
            torch.tensor([r for r in trajectory["rewards"]]).to(device).reshape(-1, 1)
        ).float() * args.reward_scale
        prompt_id_lens = trajectory["prompt_ids_lens"]
        actor_logprobs = [
            torch.tensor(lp).to(device) for lp in trajectory["action_logprobs"]
        ]
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
        ## 1) Policy log probabilities are directly from actors (vLLM).
        actor_logps = torch.zeros_like(response_masks).float()
        for i in range(len(actor_logps)):
            actor_logps[i, torch.where(response_masks[i])[0]] = actor_logprobs[i]
        ## 2) Reevaluate log probabilities using learner model.
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

                batch_logps, _ = self.get_batch_logps(
                    model=self.model,
                    input_ids=mb_input_ids,
                    att_mask=mb_att_mask,
                    temperature=args.temperature,
                    compute_entropy=False,
                )
                logps[mini_batch_inds, : mb_last_valid_token_pos - 1] = batch_logps

        ## 2) Reference.
        if self.ref_model is not None:
            all_ref_logps = []
            with torch.no_grad():
                for i in range(0, len(input_ids), args.train_batch_size_per_device):
                    batch_inds = torch.arange(i, i + args.train_batch_size_per_device)

                    batch_ref_logps, _ = self.get_batch_logps(
                        model=self.ref_model,
                        input_ids=input_ids[batch_inds],
                        att_mask=att_mask[batch_inds],
                        temperature=args.temperature,
                        compute_entropy=False,
                    )
                    all_ref_logps.append(batch_ref_logps)
            ref_logps = torch.cat(all_ref_logps)

            # Not using KL as rewards.
            # kl_rewards = -args.kl_penalty_coef * (logps - ref_logps) * response_masks
            # rewards = kl_rewards.clone()
            rewards = torch.zeros_like(response_masks).float()
            del all_ref_logps
            torch.cuda.empty_cache()
            gc.collect()
        else:
            rewards = torch.zeros_like(response_masks).float()

        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()

        advantages = torch.tensor(trajectory["advantages"]).to(device).reshape(-1, 1)
        if self.args.whiten_adv:
            local_sum = advantages.sum()
            local_square_sum = (advantages**2).sum()
            local_num = torch.tensor(
                [advantages.numel()], dtype=torch.float32, device=advantages.device
            )

            global_sum = self.strategy.all_reduce(local_sum, op="sum")
            global_square_sum = self.strategy.all_reduce(local_square_sum, op="sum")
            global_num = self.strategy.all_reduce(local_num, op="sum")

            mean_adv = global_sum / global_num
            std_adv = torch.sqrt(global_square_sum / global_num - mean_adv**2)
            advantages = (advantages - mean_adv) / (std_adv + 1e-9)
        if self.args.critic_type == "ppo":
            returns = torch.tensor(trajectory["returns"]).to(device).reshape(-1, 1)
            values = torch.tensor(trajectory["values"]).to(device).reshape(-1, 1)

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
                mb_state_ids = state_ids[i][None]

                mb_response_masks = response_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_actor_logps = actor_logps[mini_batch_inds]
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
                mb_actor_logps = mb_actor_logps[:, : mb_last_valid_token_pos - 1]

                if self.args.critic_type == "ppo":
                    mb_return = returns[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_values = values[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_advantage = mb_advantage[:, : mb_last_valid_token_pos - 1]

                # Policy learning.
                new_logps, entropy = self.get_batch_logps(
                    model=self.model,
                    input_ids=mb_input_ids,
                    att_mask=mb_att_mask,
                    temperature=args.temperature,
                    compute_entropy=True,
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

                    if self.args.tis_c is not None:
                        tis = torch.exp(mb_logps - mb_actor_logps).clamp(
                            max=self.args.tis_c
                        )
                        pg_loss_max *= tis

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

                with torch.no_grad():
                    entropy = masked_mean(entropy, mb_response_masks)
                    infos["entropy"] = entropy

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
                        input_ids=mb_state_ids,
                        attention_mask=torch.ones_like(mb_state_ids),
                    )[:, 0]

                    value_pred_clipped = torch.clamp(
                        value_pred,
                        mb_values - args.cliprange_value,
                        mb_values + args.cliprange_value,
                    )
                    vf_losses1 = torch.square(value_pred - mb_return)
                    vf_losses2 = torch.square(value_pred_clipped - mb_return)
                    vf_loss_max = torch.max(vf_losses1, vf_losses2)
                    print("vf_loss_max shape", vf_loss_max.shape)
                    vf_loss = 0.5 * vf_loss_max.mean()
                    critic_loss = args.vf_coef * vf_loss

                    self.strategy.backward(
                        critic_loss, self.critic, self.critic_optimizer
                    )
                    self.strategy.optimizer_step(
                        self.critic_optimizer, self.critic, self.critic_scheduler
                    )
                    infos["critic_loss"] = critic_loss.detach()
                    infos["vf_clipfrac"] = (
                        (vf_losses2 > vf_losses1).float().mean().detach()
                    )

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
