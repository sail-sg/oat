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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple

import torch

Metric = Dict[str, Any]


class DAPAlgo(Enum):
    DPO = 0
    IPO = 1
    SLiC = 2
    SimPO = 3
    BNF = 4
    LR_DPO = 5


class RLAlgo(Enum):
    PPO = 100


class SFTAlgo(Enum):
    SFT = 200


@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str
    chosen_id: int = 0
    chosen_feature: torch.Tensor = None
    rejected_feature: torch.Tensor = None
    init_clash: bool = False
    loss_mask: bool = True
    is_model_data: bool = False
    info: Metric = None


@dataclass
class TrajectoryData:
    prompt: str
    prompt_ids: List[int]
    response: str
    response_ids: List[int]
    response_logprobs: List[float]
    rewards: List[float]
    loss_mask: bool = True
    info: Metric = None


class RewardData(NamedTuple):
    pair_features: torch.Tensor  # (B, 2, d)
    loss_masks: torch.Tensor  # (B,)
