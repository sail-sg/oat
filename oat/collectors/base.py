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

import logging
import time
from itertools import accumulate
from typing import List, Union

import Levenshtein
import numpy as np
import torch
import tree
from torch.distributed import gather_object

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.types import PreferenceData, TransitionData
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
        feedback_data: List[Union[PreferenceData, TransitionData]],
    ):
        metric = {
            "actor/total_time": actor_time,
            "actor/num_transitions": len(feedback_data),
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
        elif isinstance(feedback_data[0], TransitionData):
            metric.update(
                {
                    "actor/generate_avg_str_len": np.mean(
                        [len(t.response) for t in feedback_data]
                    ),
                    "actor/response_tok_len": np.mean(
                        [len(t.response_ids) for t in feedback_data]
                    ),
                    "actor/avg_reward": np.mean(
                        [max(t.rewards) for t in feedback_data]
                    ),
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
        prompts: List[str],
        formatted_prompts: List[str],
        refs: List[str],
        same_actor_group: torch.distributed.ProcessGroup,
    ):
        # generate response & get feedback
        st_time = time.time()

        rank = torch.distributed.get_rank()
        actor = self.actors[(rank // self.args.num_gpus_per_actor) % len(self.actors)]
        logging.info(
            f"Learner {rank} local actor rank: {(rank // self.args.num_gpus_per_actor) % len(self.actors)}"
        )

        # allgather the arguments and only invoke step on the first rank in the same actor group
        def prepare_gather_list():
            return (
                [None] * torch.distributed.get_world_size(same_actor_group)
                if torch.distributed.get_rank(same_actor_group) == 0
                else None
            )

        gathered_prompts = prepare_gather_list()
        gathered_formatted_prompts = prepare_gather_list()
        gathered_refs = prepare_gather_list()
        logging.info(
            f"Learner {rank} prompts size: {len(prompts)}, formatted_prompts size: {len(formatted_prompts)}, refs size: {len(refs)}"
        )
        gather_object(prompts, gathered_prompts, group=same_actor_group, group_dst=0)
        gather_object(
            formatted_prompts,
            gathered_formatted_prompts,
            group=same_actor_group,
            group_dst=0,
        )
        gather_object(refs, gathered_refs, group=same_actor_group, group_dst=0)
        if torch.distributed.get_rank(same_actor_group) == 0:
            prompts = [item for sublist in gathered_prompts for item in sublist]
            formatted_prompts = [
                item for sublist in gathered_formatted_prompts for item in sublist
            ]
            if refs is not None:
                refs = [item for sublist in gathered_refs for item in sublist]

            logging.info(f"rank {rank} invoking step on actor {actor}")
            if self.args.online_evaluation:
                handle = actor.step(prompts, formatted_prompts, refs)
            else:
                handle = actor.step(prompts, formatted_prompts)
            feedback_data: List[Union[PreferenceData, TransitionData]] = (
                self.ipc_client.deserialize_ipc(handle)
            )
            logging.info(
                f"Group Leader Learner {rank} feedback_data size: {len(feedback_data)}"
            )
            assert len(feedback_data) % len(prompts) == 0
            sample_per_prompt = len(feedback_data) // len(prompts)
            rank_lengths = [
                len(sublist) * sample_per_prompt for sublist in gathered_prompts
            ]
            assert len(feedback_data) == sum(rank_lengths)
            feedback_data = [
                feedback_data[i:j]
                for i, j in zip(
                    [0] + list(accumulate(rank_lengths)), accumulate(rank_lengths)
                )
            ]
        else:
            feedback_data = None

        scattered_feedback_data = [None]
        torch.distributed.scatter_object_list(
            scattered_feedback_data, feedback_data, group=same_actor_group, group_src=0
        )
        scattered_feedback_data = scattered_feedback_data[0]
        logging.info(
            f"Learner {rank} scattered_feedback_data size: {len(scattered_feedback_data)}"
        )

        actor_time = time.time() - st_time
        return scattered_feedback_data, self.get_metrics(
            actor_time, scattered_feedback_data
        )
