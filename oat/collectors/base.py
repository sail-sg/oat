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

from oat.types import PreferenceData


class PreferenceCollector:
    def __init__(self, learner) -> None:
        self.learner = learner

    def get_metrics(self, actor_time, preference_data):
        metric = {
            "actor/total_time": actor_time,
            "actor/chosen_avg_str_len": np.mean(
                [len(p.chosen_response) for p in preference_data]
            ),
            "actor/rejected_avg_str_len": np.mean(
                [len(p.rejected_response) for p in preference_data]
            ),
            "actor/init_clash_ratio": np.mean([p.init_clash for p in preference_data]),
            "actor/same_response_ratio": np.mean([p.same for p in preference_data]),
            "actor/pair_edit_dist": np.mean(
                [
                    Levenshtein.distance(p.chosen_response, p.rejected_response)
                    for p in preference_data
                ]
            ),
            "actor/chosen_id": np.mean([p.chosen_id for p in preference_data]),
        }

        mean_info = tree.map_structure(
            lambda *x: np.mean(x), *[p.info for p in preference_data]
        )
        metric.update(mean_info)

        return metric

    def collect_preference(
        self,
        prompts: Union[str, List[str]],
        formatted_prompts: List[str],
        refs: Union[str, List[str]],
    ):
        # generate response & get feedback
        st_time = time.time()

        rank = torch.distributed.get_rank()
        actor = self.learner.actors[rank % len(self.learner.actors)]
        if self.learner.strategy.args.online_evaluation:
            handle = actor.step(prompts, formatted_prompts, refs)
        else:
            handle = actor.step(prompts, formatted_prompts)
        preference_data: List[PreferenceData] = self.learner.ipc_client.deserialize_ipc(
            handle
        )

        actor_time = time.time() - st_time
        return preference_data, self.get_metrics(actor_time, preference_data)
