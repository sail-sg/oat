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

import numpy as np


def compute_lambda_returns(
    r_t: np.ndarray, discount_t: np.ndarray, v_t: np.ndarray, lambda_: np.ndarray = 1.0
):
    """Estimates a multistep truncated lambda return from a trajectory (rewritten from rlax).

    Given a a trajectory of length `T+1`, generated under some policy π, for each
    time-step `t` we can estimate a target return `G_t`, by combining rewards,
    discounts, and state values, according to a mixing parameter `lambda`.

    The parameter `lambda_`  mixes the different multi-step bootstrapped returns,
    corresponding to accumulating `k` rewards and then bootstrapping using `v_t`.

        rₜ₊₁ + γₜ₊₁ vₜ₊₁
        rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ vₜ₊₂
        rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ rₜ₊₂ + γₜ₊₁ γₜ₊₂ γₜ₊₃ vₜ₊₃

    The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:

        Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].

    In the `on-policy` case, we estimate a return target `G_t` for the same
    policy π that was used to generate the trajectory. In this setting the
    parameter `lambda_` is typically a fixed scalar factor. Depending
    on how values `v_t` are computed, this function can be used to construct
    targets for different multistep reinforcement learning updates:

        TD(λ):  `v_t` contains the state value estimates for each state under π.
        Q(λ):  `v_t = max(q_t, axis=-1)`, where `q_t` estimates the action values.
        Sarsa(λ):  `v_t = q_t[..., a_t]`, where `q_t` estimates the action values.

    In the `off-policy` case, the mixing factor is a function of state, and
    different definitions of `lambda` implement different off-policy corrections:

        Per-decision importance sampling:  λₜ = λ ρₜ = λ [π(aₜ|sₜ) / μ(aₜ|sₜ)]
        V-trace, as instantiated in IMPALA:  λₜ = min(1, ρₜ)

    Note that the second option is equivalent to applying per-decision importance
    sampling, but using an adaptive λ(ρₜ) = min(1/ρₜ, 1), such that the effective
    bootstrap parameter at time t becomes λₜ = λ(ρₜ) * ρₜ = min(1, ρₜ).
    This is the interpretation used in the ABQ(ζ) algorithm (Mahmood 2017).

    Of course this can be augmented to include an additional factor λ.  For
    instance we could use V-trace with a fixed additional parameter λ = 0.9, by
    setting λₜ = 0.9 * min(1, ρₜ) or, alternatively (but not equivalently),
    λₜ = min(0.9, ρₜ).

    Estimated return are then often used to define a td error, e.g.:  ρₜ(Gₜ - vₜ).

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/sutton/book/ebook/node74.html).

    Args:
        r_t: sequence of rewards rₜ for timesteps t in [1, T].
        discount_t: sequence of discounts γₜ for timesteps t in [1, T].
        v_t: sequence of state values estimates under π for timesteps t in [1, T].
        lambda_: mixing parameter; a scalar or a vector for timesteps t in [1, T].

    Returns:
        Multistep lambda returns.
    """
    lambda_ = np.ones_like(r_t) * lambda_

    def _body(acc, xs):
        returns, discounts, values, lambda_ = xs
        acc = returns + discounts * ((1 - lambda_) * values + lambda_ * acc)
        return acc, acc

    carry = v_t[-1]
    ys = [None] * len(r_t)
    for t in reversed(range(len(r_t))):
        carry, y = _body(carry, (r_t[t], discount_t[t], v_t[t], lambda_[t]))
        ys[t] = y

    return np.array(ys)
