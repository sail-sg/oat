import functools
from multiprocessing import Pool, TimeoutError
from typing import Any, List, Tuple

import torch

from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric
from oat.utils.math_grader import (
    answer_tag_reward_fn,
    boxed_reward_fn,
    r1_distill_qwen_math_reward_fn,
)


class MATHOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the math answer grading."""

    def __init__(
        self, template, verifier_version, correct_reward, incorrect_reward
    ) -> None:
        super().__init__()
        if template == "r1":
            math_reward_fn = answer_tag_reward_fn
        elif template == "r1_distill_qwen":
            math_reward_fn = r1_distill_qwen_math_reward_fn
        else:
            math_reward_fn = boxed_reward_fn

        self.math_reward_fn = functools.partial(
            math_reward_fn,
            fast=verifier_version == "fast",
            correct_reward=correct_reward,
            incorrect_reward=incorrect_reward,
        )
        self.incorrect_reward = incorrect_reward
        # Process pool is used to enable the timeout mechanism for answer grading in our distributed training setup.
        self.mp_pool = Pool(2)

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        # Parameters used by Oat when using model-based reward, here we don't need.
        del inputs, batch_size

        rewards = []
        infos = []
        for resp, ref in zip(responses, references):
            res = self.mp_pool.apply_async(self.math_reward_fn, (resp, ref))
            try:
                info, r = res.get(timeout=1)
                rewards.append(r)
                infos.append(info)
            except TimeoutError:
                rewards.append(self.incorrect_reward)
                infos.append({"formatted": False})

        return torch.tensor(rewards), infos

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info
