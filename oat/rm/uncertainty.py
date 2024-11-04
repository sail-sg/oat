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

import einops
import torch


def kl_divergence(rewards: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Epistemic uncertainty measured by model disagreement between ensembles.

    Calculates KL divergence between individual distribution predictions and
    the Bernoulli mixture distribution.

    OATArgs:
        rewards (torch.Tensor): Reward prediction (logits), (E, M, N, 1)

    Returns:
        torch.Tensor: Uncertainty, (M, N, N')
    """
    E = rewards.shape[0]
    p = bradley_terry_prob_with_temp(
        rewards,
        einops.rearrange(rewards, "e m n 1 -> e m 1 n"),
        temperature=temperature,
    )  # (E, M, N, N')

    p_mean = p.mean(dim=0, keepdim=True)
    pc = 1 - p
    pc_mean = 1 - p_mean

    component_p = p * torch.log(p / p_mean)
    component_pc = pc * torch.log(pc / pc_mean)

    repeat_p_mean = p_mean.repeat(E, 1, 1, 1)
    repeat_pc_mean = pc_mean.repeat(E, 1, 1, 1)
    kl = torch.where(
        repeat_p_mean == 1,
        component_p,
        torch.where(repeat_pc_mean == 1, component_pc, component_p + component_pc),
    ).mean(dim=0)

    # Avoid numerical errors.
    nan_idx = torch.isnan(kl)
    kl_T = kl.transpose(-1, -2)
    kl[nan_idx] = kl_T[nan_idx]
    return (kl + kl_T) / 2


def logits_variance(rewards: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Computes variance of pre-sigmoid logits."""
    del temperature
    pref_logits = rewards - einops.rearrange(
        rewards, "e m n 1 -> e m 1 n"
    )  # (E, M, N, N')
    return pref_logits.var(dim=0)


def probabilities_variance(
    rewards: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    prob = bradley_terry_prob_with_temp(
        rewards,
        einops.rearrange(rewards, "e m n 1 -> e m 1 n"),
        temperature=temperature,
    )
    return prob.var(dim=0)


def bernoulli_variance(rewards: torch.Tensor):
    prob = bradley_terry_prob_with_temp(
        rewards,
        einops.rearrange(rewards, "e m n 1 -> e m 1 n"),
        temperature=1.0,
    ).mean(0)
    return prob * (1 - prob)


def bradley_terry_prob_with_temp(scores_1, score_2, temperature=1.0):
    return 1 / (1 + torch.exp(-(scores_1 - score_2) / temperature))
