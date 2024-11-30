# Copyright 2024 Garena Online Private Limited
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

import abc
import random
from typing import Any, Dict, Tuple

import einops
import torch
import torch.nn.functional as F
from torch import nn, optim

from oat.args import OATArgs
from oat.rm import uncertainty
from oat.rm.networks import EnsembleModel
from oat.utils.buffer import UniformBuffer


class RewardModel(abc.ABC, nn.Module):

    train_bs = 128
    infer_bs = 128

    @abc.abstractclassmethod
    def get_metrics(cls):
        """Get learning metrics."""

    @abc.abstractmethod
    def get_duel_actions(
        self, features: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Get dueling actions based on rewards of given features.

        Args:
            features (torch.Tensor): (M, N, d)

        Returns:
            Tuple[torch.LongTensor]: rewards, first and second indices [(E or 2, M, N, 1), (M, 1), (M, 1)]
        """

    @abc.abstractmethod
    def get_best_action(self, features: torch.Tensor) -> torch.LongTensor:
        """Get Best-of-N action based on rewards of given features.

        Args:
            features (torch.Tensor): (M, N, d)

        Returns:
            torch.LongTensor: (M, 1)
        """

    @abc.abstractmethod
    def learn(self, buffer: UniformBuffer) -> Dict[str, Any]:
        """Learn the reward model based on preference data."""

    @abc.abstractmethod
    def get_rewards(self, features: torch.Tensor) -> torch.Tensor:
        """Compute rewards."""


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self,
        chosen_reward: torch.Tensor,
        reject_reward: torch.Tensor,
        mask: torch.Tensor,
        margin: torch.Tensor = None,
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return (loss * mask).mean()


class EnnEETS(RewardModel):
    """E&E Thompson Sampling based on ensemble."""

    @classmethod
    def get_metrics(cls):
        return {
            "train/rm/loss_rew": 0,
            "train/rm/loss_reg": 0,
            "train/rm/chosen_rewards": 0,
            "train/rm/rejected_rewards": 0,
            "train/rm/lambda": 0,
        }

    def __init__(self, args: OATArgs) -> None:
        super().__init__()
        assert args.enn_max_try <= args.num_ensemble

        self.model = EnsembleModel(
            encoding_dim=getattr(
                args, "encoding_dim", 2048
            ),  # Fixed due to PairRM's backbone
            num_ensemble=args.num_ensemble,
            hidden_dim=args.rm_hidden_dim,
            activation=args.rm_act_fn,
        )
        self.model.init()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.rm_lr, weight_decay=args.rm_wd
        )
        self.reg_lambda = args.enn_lambda
        self.max_resample = args.enn_max_try
        self.allow_second_best = args.exp_allow_second_best
        self.sgd_steps = args.rm_sgd_steps
        self.loss_fn = PairWiseLoss()

    @torch.no_grad
    def get_rewards(self, features: torch.Tensor) -> torch.Tensor:
        M, N, _ = features.shape
        E = self.model.num_ensemble
        features = einops.rearrange(features, "m n d -> (m n) d")
        rewards = []
        for ndx in range(0, len(features), self.infer_bs):
            batch_feat = features[ndx : min(ndx + self.infer_bs, len(features))]
            batch_feat = batch_feat[None, :, :].repeat([E, 1, 1])
            rewards.append(self.model(batch_feat))
        rewards = torch.cat(rewards, dim=1)  # (E, M*N, 1)
        rewards = rewards.view(E, M, N, 1)
        return rewards

    @torch.no_grad
    def get_best_action(self, features: torch.Tensor) -> torch.LongTensor:
        rewards = self.get_rewards(features)  # (E, M, N, 1)
        avg_rewards = rewards.mean(0)  # (M, N, 1)
        best_actions = avg_rewards.argmax(dim=1)  # (M, 1)
        return best_actions

    @torch.no_grad
    def get_duel_actions(
        self, features: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        rewards = self.get_rewards(features)
        E = rewards.shape[0]
        best_actions = rewards.argmax(dim=2)  # (E, M, 1)
        # sample without replacement
        s1 = list(range(E))
        random.shuffle(s1)
        first_actions = best_actions[s1[0]]
        second_actions = torch.ones_like(first_actions) * -1
        for actions in best_actions[s1[1 : self.max_resample]]:
            valid_idx = (actions != first_actions) * (second_actions == -1)
            second_actions[valid_idx] = actions[valid_idx]
            if -1 not in second_actions:
                break
        if self.allow_second_best:
            second_best_actions = rewards.argsort(dim=2)[..., -2, :]
            for actions in second_best_actions[s1[: self.max_resample]]:
                valid_idx = (actions != first_actions) * (second_actions == -1)
                second_actions[valid_idx] = actions[valid_idx]
                if -1 not in second_actions:
                    break
        second_actions = torch.where(
            second_actions == -1, first_actions, second_actions
        )
        return rewards, first_actions, second_actions

    def learn(self, buffer: UniformBuffer) -> Dict[str, Any]:
        total_num_queries = buffer.total_num_queries
        for _ in range(self.sgd_steps):
            batch = buffer.sample(self.train_bs)
            if batch is None:
                return self.get_metrics()
            pair_feats = batch.pair_features.view(2 * self.train_bs, -1)
            batch_inp = pair_feats[None, :, :].repeat([self.model.num_ensemble, 1, 1])
            scores = self.model(batch_inp)
            scores = scores.view(self.model.num_ensemble, self.train_bs, 2, 1)
            chosen_scores, rejected_scores = scores[..., 0, :], scores[..., 1, :]
            loss_rew = self.loss_fn(
                chosen_scores, rejected_scores, batch.loss_masks[None]
            )
            loss_reg = (
                self.reg_lambda
                * self.train_bs
                / total_num_queries
                * self.model.regularization()
            )
            self.optimizer.zero_grad()
            (loss_rew + loss_reg).backward()
            self.optimizer.step()

        return {
            "train/rm/loss_rew": loss_rew.detach(),
            "train/rm/loss_reg": loss_reg.detach(),
            "train/rm/chosen_rewards": chosen_scores.mean().detach(),
            "train/rm/rejected_rewards": rejected_scores.mean().detach(),
            "train/rm/lambda": self.reg_lambda * self.train_bs / total_num_queries,
        }


class EnnUncertainty(EnnEETS):
    """Pure exploration based on ensemble."""

    def __init__(self, args: OATArgs) -> None:
        super().__init__(args)
        self.uct_fn = uncertainty.logits_variance

    @torch.no_grad
    def get_duel_actions(self, features: torch.Tensor) -> Tuple[torch.LongTensor]:
        rewards = self.get_rewards(features)  # (E, M, N, 1)
        _, M, N, _ = rewards.shape
        pref_uncertainty = self.uct_fn(rewards)
        flatten_idx = pref_uncertainty.view(M, -1).argmax(-1)
        first_actions = flatten_idx // N
        second_actions = flatten_idx % N
        return rewards, first_actions.view(M, 1), second_actions.view(M, 1)


class EnnBAITS(EnnEETS):
    """BAI Thompson Sampling based on ensemble."""

    def __init__(self, args: OATArgs) -> None:
        super().__init__(args)
        self.uct_fn = uncertainty.logits_variance

    @torch.no_grad
    def get_duel_actions(
        self, features: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        rewards = self.get_rewards(features)  # (E, M, N, 1)
        E, M, _, _ = rewards.shape
        best_actions = rewards.argmax(dim=2)  # (E, M, 1)
        # sample without replacement
        s1 = list(range(E))
        random.shuffle(s1)
        first_actions = best_actions[s1[0]]

        pref_uncertainty = self.uct_fn(rewards)

        second_actions = torch.stack(
            [pref_uncertainty[i][first_actions[i]].argmax() for i in range(M)], dim=0
        ).view(M, 1)

        return rewards, first_actions, second_actions


class EnnPassive(EnnEETS):
    """Learning RM but not for sampling, only for BoN generation."""

    @torch.no_grad
    def get_duel_actions(self, features: torch.Tensor) -> Tuple[torch.LongTensor]:
        M, _, _ = features.shape  # M, N, d
        rewards = self.get_rewards(features)
        first_actions = torch.zeros((M, 1), device=features.device).long()
        second_actions = torch.ones((M, 1), device=features.device).long()
        return rewards, first_actions, second_actions


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise
