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

import torch

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.collectors.base import FeedbackCollector
from oat.types import PreferenceData, TrajectoryData
from oat.utils.ipc import PlasmaShmClient


class AsyncFeedbackCollector(FeedbackCollector):
    def __init__(
        self, args: OATArgs, actors: List[ActorBase], ipc_client: PlasmaShmClient
    ) -> None:
        self.args = args
        self.actors = actors
        self.ipc_client = ipc_client
        self.prev_fut = None

    def collect_feedback(
        self,
        prompts: Union[str, List[str]],
        formatted_prompts: List[str],
        refs: Union[str, List[str]],
    ):
        # generate response & get feedback
        st_time = time.time()

        if self.prev_fut is not None:
            handle = self.prev_fut.result()
            feedback_data: List[Union[PreferenceData, TrajectoryData]] = (
                self.ipc_client.deserialize_ipc(handle)
            )
        else:
            feedback_data = None

        rank = torch.distributed.get_rank()
        actor = self.actors[rank % len(self.actors)]
        if self.args.online_evaluation:
            handle_fut = actor.futures.step(prompts, formatted_prompts, refs)
        else:
            handle_fut = actor.futures.step(prompts, formatted_prompts)

        self.prev_fut = handle_fut

        actor_time = time.time() - st_time

        if feedback_data is not None:
            metrics = self.get_metrics(actor_time, feedback_data)
        else:
            metrics = {}

        return feedback_data, metrics
