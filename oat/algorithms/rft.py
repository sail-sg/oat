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

import itertools
import time
from typing import List

import pandas as pd
import tree

from oat.actors import PreferenceActor
from oat.learners import SFTLearner
from oat.types import PreferenceData


class RESTLearner(SFTLearner):
    """Simply SFT."""


class RESTActor(PreferenceActor):
    """Inherit PreferenceActor but we only make use of `chosen`."""

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[PreferenceData]:
        assert not self.eval_mode
        info = {}

        # step 1. generate
        st = time.time()
        all_candidates = self.generate(formatted_prompts, self.sampling_params)
        info["actor/generate_time"] = time.time() - st

        flatten_prompts = list(
            itertools.chain.from_iterable(
                itertools.repeat(x, self.sampling_params.n) for x in prompts
            )
        )
        flatten_responses = tree.flatten(all_candidates)
        flatten_references = list(
            itertools.chain.from_iterable(
                itertools.repeat(x, self.sampling_params.n) for x in references
            )
        )

        # step 2. verify
        flatten_rewards, oracle_info = self.oracle.get_reward(
            flatten_prompts,
            flatten_responses,
            flatten_references,
        )
        rewards = flatten_rewards.reshape(len(prompts), self.sampling_params.n)

        info["actor/pass@k"] = (rewards.sum(-1) > 0).float().mean().item()
        info["actor/rewards_mean"] = rewards.mean().item()
        info["actor/oracle_time"] = time.time() - st

        # step 3. filter
        trajectories = {"prompt": [], "chosen": []}
        for x, y, is_correct in zip(
            flatten_prompts, flatten_responses, flatten_rewards
        ):
            if is_correct == 1:
                trajectories["prompt"].append(x)
                trajectories["chosen"].append(y)

        # Drop duplicates based on exact match.
        # Fancier methods (e.g., similarity-based) can be done here.
        trajectories = (
            pd.DataFrame(trajectories).drop_duplicates(ignore_index=True).to_dict()
        )
        unique_count = len(trajectories["prompt"])

        info["actor/unique_trajectories"] = unique_count

        filtered_data = []
        for i in range(unique_count):
            filtered_data.append(
                PreferenceData(
                    prompt=trajectories["prompt"][i],
                    chosen_response=trajectories["chosen"][i],
                    rejected_response="",  # Nothing.
                    info=info,
                )
            )

        handle = self.ipc_client.serialize_ipc(filtered_data)
        return handle
