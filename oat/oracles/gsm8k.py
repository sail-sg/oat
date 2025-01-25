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

from typing import Any, List, Optional, Tuple

import regex as re
import torch

from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric

FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
)


class GSM8KOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the GSM8K task."""

    def __init__(self, use_original_format: bool = False, **_) -> None:
        super().__init__()
        self.use_original_format = use_original_format

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        del inputs, batch_size
        predicted_answers = []
        rewards = []

        for resp, ref in zip(responses, references):
            answer_candidate = self._extract_predicted_answer_from_text(resp)
            predicted_answers.append(answer_candidate)
            grading_res = self._grade_answer(answer_candidate, ref)
            rewards.append(float(grading_res))

        return torch.tensor(rewards), {"predicted_answers": predicted_answers}

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

    def _extract_predicted_answer_from_text(self, text: str) -> Optional[str]:
        if self.use_original_format:
            # Extract the final answer based on ####
            if "####" not in text:
                return None
            parts = text.split("####")
            assert len(parts) >= 2
            return parts[-1].strip()

        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
        if len(pred_answer) == 0:
            return None
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip().rstrip(".")
            return pred_answer

    def _grade_answer(self, pred_answer: str, gt_answer: str) -> bool:
        if pred_answer is None:
            return False
        return (
            pred_answer.strip().replace(",", "").lower()
            == gt_answer.replace(",", "").strip().lower()
        )
