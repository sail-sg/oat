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
from typing import List

import torch

from oat.collectors.base import PreferenceCollector
from oat.types import PreferenceData


class AsyncPreferenceCollector(PreferenceCollector):
    def __init__(self, learner) -> None:
        super().__init__(learner)
        self.prev_fut = None

    def collect_preference(
        self,
        prompts: str | List[str],
        formatted_prompts: List[str],
        refs: str | List[str],
    ):
        # generate response & get feedback
        st_time = time.time()

        if self.prev_fut is not None:
            handle = self.prev_fut.result()
            preference_data: List[PreferenceData] = (
                self.learner.ipc_client.deserialize_ipc(handle)
            )
        else:
            preference_data = None

        rank = torch.distributed.get_rank()
        actor = self.learner.actors[rank % len(self.learner.actors)]
        if self.learner.strategy.args.online_evaluation:
            handle_fut = actor.futures.step(prompts, formatted_prompts, refs)
        else:
            handle_fut = actor.futures.step(prompts, formatted_prompts)

        self.prev_fut = handle_fut

        actor_time = time.time() - st_time

        if preference_data is not None:
            metrics = self.get_metrics(actor_time, preference_data)
        else:
            metrics = {}

        return preference_data, metrics
