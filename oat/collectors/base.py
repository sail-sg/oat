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

import time
from typing import List, Union

import Levenshtein
import numpy as np
import torch
import tree

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.types import PreferenceData, TrajectoryData
from oat.utils.ipc import PlasmaShmClient


class FeedbackCollector:
    def __init__(
        self, args: OATArgs, actors: List[ActorBase], ipc_client: PlasmaShmClient
    ) -> None:
        self.args = args
        self.actors = actors
        self.ipc_client = ipc_client

    def get_metrics(
        self,
        actor_time: float,
        feedback_data: List[Union[PreferenceData, TrajectoryData]],
    ):
        metric = {
            "actor/total_time": actor_time,
        }
        if isinstance(feedback_data[0], PreferenceData):
            metric.update(
                {
                    "actor/chosen_avg_str_len": np.mean(
                        [len(p.chosen_response) for p in feedback_data]
                    ),
                    "actor/rejected_avg_str_len": np.mean(
                        [len(p.rejected_response) for p in feedback_data]
                    ),
                    "actor/init_clash_ratio": np.mean(
                        [p.init_clash for p in feedback_data]
                    ),
                    "actor/loss_mask": np.mean([p.loss_mask for p in feedback_data]),
                    "actor/pair_edit_dist": np.mean(
                        [
                            Levenshtein.distance(p.chosen_response, p.rejected_response)
                            for p in feedback_data
                        ]
                    ),
                    "actor/chosen_id": np.mean([p.chosen_id for p in feedback_data]),
                }
            )
        elif isinstance(feedback_data[0], TrajectoryData):
            metric.update(
                {
                    "actor/generate_avg_str_len": np.mean(
                        [len(t.response) for t in feedback_data]
                    )
                }
            )
        else:
            raise ValueError("Invalid feedback data type.")

        mean_info = tree.map_structure(
            lambda *x: np.mean(x), *[p.info for p in feedback_data]
        )
        metric.update(mean_info)

        return metric

    def collect_feedback(
        self,
        prompts: Union[str, List[str]],
        formatted_prompts: List[str],
        refs: Union[str, List[str]],
    ):
        # generate response & get feedback
        st_time = time.time()

        rank = torch.distributed.get_rank()
        actor = self.actors[rank % len(self.actors)]
        if self.args.online_evaluation:
            handle = actor.step(prompts, formatted_prompts, refs)
        else:
            handle = actor.step(prompts, formatted_prompts)
        feedback_data: List[Union[PreferenceData, TrajectoryData]] = (
            self.ipc_client.deserialize_ipc(handle)
        )

        actor_time = time.time() - st_time
        return feedback_data, self.get_metrics(actor_time, feedback_data)
