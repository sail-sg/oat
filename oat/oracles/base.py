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
from typing import Any, List, Tuple

import torch

from oat.types import Metric


class PreferenceOracleBase(abc.ABC):
    @abc.abstractmethod
    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Provide oracle preference feedback.

        Args:
            inputs (List[str]): List of input strings.
            candidates_A (List[str]): List of candidate A strings.
            candidates_B (List[str]): List of candidate B strings
            batch_size (int, optional): Batch size. Defaults to 4.
            disable_tqdm (bool, optional): Print progress. Defaults to False.

        Returns:
            List[Any]:
                - List[float], logits as confidence that A is better than B.
                    >0 means A is better than B, <0 means B is better than A
                - List[bool], True if A is better than B, False otherwise
            Metric: Extra information from the oracle.
        """


class RewardOracleBase(abc.ABC):
    @abc.abstractmethod
    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        """Provide oracle reward feedback.

        Args:
            inputs (List[str]): List of input strings.
            responses (List[str]): List of response strings.
            references (List[str]): List of references strings.
            batch_size (int, optional): Batch size. Defaults to 4.

        Returns:
            torch.Tensor: Rewards.
            Metric: Extra information from the oracle.
        """
