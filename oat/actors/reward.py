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
import logging
import time
from typing import List

import tree

from oat.actors.base import ActorBase
from oat.types import TrajectoryData


class RewardActor(ActorBase):
    """The environment is a reward oracle. In this case the problem can be formulated
    as conventional reinforcement learning or contextual bandit.

    When the reward is a trained model from human preferences, this is also known as RLHF.
    """

    def init(self, actor_id):
        super().init(actor_id)
        # Return the log prob because RL algos need it.
        self.sampling_params.logprobs = 1

    def extract_candidates_from_output(self, outputs, sampling_params, strip=True):
        candidates = []
        for i in range(len(outputs)):
            # for each prompt
            candidates.append([])
            for k in range(sampling_params.n):
                # for each response
                text = outputs[i].outputs[k].text
                if strip:
                    text = text.strip()
                candidates[i].append(text)
        return candidates

    def generate_and_maybe_eval(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ):
        assert self.eval_mode
        outputs = self.generate(formatted_prompts, self.eval_sampling_params)
        candidates = self.extract_candidates_from_output(
            outputs, self.eval_sampling_params
        )
        responses = []
        for j in range(self.eval_sampling_params.n):
            responses.extend([candidates[i][j] for i in range(len(prompts))])

        win_probs = None
        if references:
            logging.debug(f"Evaluating using oracle {self.oracle}")
            st = time.time()
            win_probs, _ = self.oracle.compare(
                prompts * self.eval_sampling_params.n,
                responses,
                references * self.eval_sampling_params.n,
                batch_size=self.oracle_batch_size,
                return_probs=True,
                disable_tqdm=True,
            )
            logging.debug(f"Time elapse {time.time() - st}")
        reshaped_responses = []
        for x_i in range(len(prompts)):
            reshaped_responses.append(
                [responses[y_i] for y_i in range(x_i, len(responses), len(prompts))]
            )
        reshaped_win_probs = win_probs.reshape(
            self.eval_sampling_params.n, len(prompts)
        ).transpose(1, 0)
        return reshaped_responses, reshaped_win_probs

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[TrajectoryData]:
        assert not self.eval_mode
        info = {}

        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)
        all_candidates = self.extract_candidates_from_output(
            outputs, self.sampling_params
        )
        info["actor/generate_time"] = time.time() - st

        # step 2. query for oracle reward
        st = time.time()

        rewards, oracle_info = self.oracle.get_reward(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in prompts
                )
            ),
            tree.flatten(all_candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in references
                )
            ),
        )
        rewards = rewards.reshape(len(prompts), self.sampling_params.n)

        info["actor/rewards_mean"] = rewards.mean().item()
        info["actor/rewards_std"] = rewards.std().item()
        info["actor/rewards_std_per_prompt"] = rewards.std(1).mean().item()
        info["actor/oracle_time"] = time.time() - st
        # info.update({f"oracle/{k}": v for k, v in oracle_info.items()})

        trajectory_data = [
            TrajectoryData(
                prompt=prompts[i],
                responses=all_candidates[i],
                rewards=rewards[i],
                info=info,
            )
            for i in range(len(prompts))
        ]

        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle
