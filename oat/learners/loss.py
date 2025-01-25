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

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from oat.types import DAPAlgo


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(
        self,
        beta: float,
        label_smoothing: float = 0.0,
        len_reg_alpha: float = 0.0,
        dpo_positive_lambda: float = 0.0,
        sft_weight: float = 0.0,
        dap_algo=DAPAlgo.DPO,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.dpo_positive_lambda = dpo_positive_lambda
        self.sft_weight = sft_weight
        self.len_reg_alpha = len_reg_alpha
        self.dap_algo = dap_algo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        loss_masks: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.dap_algo == DAPAlgo.IPO:
            losses = (
                logits - 1 / (2 * self.beta)
            ) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        elif self.dap_algo == DAPAlgo.SLiC:
            losses = torch.relu(1 - self.beta * logits)
        else:
            if self.len_reg_alpha > 0:
                y_length = token_masks.sum(-1)
                length_diff = (
                    y_length[: len(y_length) // 2] - y_length[len(y_length) // 2 :]
                )
                # Eq. 9 https://arxiv.org/pdf/2403.19159; Length Reg in loss.
                logits += self.len_reg_alpha / self.beta * length_diff

            if self.dpo_positive_lambda > 0:
                # Eq. 3 https://arxiv.org/pdf/2402.13228; mitigates chosen prob decreasing issue.
                logits -= self.dpo_positive_lambda * torch.relu(
                    reference_chosen_logps - policy_chosen_logps
                )
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            # Add SFT loss.
            if self.sft_weight > 0:
                losses -= (
                    self.sft_weight
                    * policy_chosen_logps
                    / token_masks.sum(-1)[: (len(token_masks) // 2)]
                )

        loss = (losses * loss_masks).mean()
        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return loss, chosen_rewards, rejected_rewards


class SimPOLoss(nn.Module):
    def __init__(
        self,
        beta: float,
        gamma_beta_ratio: float,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.gamma_beta_ratio = gamma_beta_ratio
        self.loss_type = loss_type
        assert loss_type in (
            "sigmoid",
            "hinge",
        ), f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        loss_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        logits = pi_logratios - self.gamma_beta_ratio
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise RuntimeError

        loss = (losses * loss_masks).mean()
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return loss, chosen_rewards, rejected_rewards


class BNFLoss(nn.Module):
    """
    BNF Loss
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        policy_logps: torch.Tensor,
        policy_entropy: torch.Tensor,
        ref_logps: torch.Tensor,
        token_masks: torch.Tensor,
        loss_masks: torch.Tensor,
        chosen_shape: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        token_count = torch.min(token_masks.sum(-1, keepdims=True))
        target_response = torch.clamp(
            policy_logps.exp() / (ref_logps.exp() + 1e-6),
            max=torch.ones_like(policy_logps),
        )
        target_other = (1 - target_response + 1e-6) / (1 - policy_logps.exp() + 1e-6)
        logp = (
            target_response.detach() * policy_logps
            + target_other.detach() * policy_entropy
        )

        logp = logp * token_masks
        logp_sum = logp.sum(-1, keepdims=True) / token_count

        rewards = policy_logps.sum(-1, keepdims=True) - ref_logps.sum(-1, keepdims=True)

        chosen_logp_sum = logp_sum[: chosen_shape[0]]
        rejected_logp_sum = logp_sum[chosen_shape[0] :]

        chosen_rewards = rewards[: chosen_shape[0]]
        rejected_rewards = rewards[chosen_shape[0] :]

        losses = -(chosen_logp_sum - rejected_logp_sum)

        loss = (losses * loss_masks).mean()

        return loss, chosen_rewards, rejected_rewards
