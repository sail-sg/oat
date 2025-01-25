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

from typing import Any, List, Tuple

import llm_blender
import torch

from oat.oracles.base import PreferenceOracleBase
from oat.types import Metric


class PairRMOracle(PreferenceOracleBase):
    def __init__(self, **_) -> None:
        super().__init__()
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        logits = self.blender.compare(
            inputs,
            candidates_A,
            candidates_B,
            batch_size=batch_size,
            return_logits=True,
            disable_tqdm=disable_tqdm,
        )
        probs = torch.from_numpy(logits).sigmoid().numpy()
        if return_probs:
            return probs, {}
        else:
            return probs > 0.5, {}
