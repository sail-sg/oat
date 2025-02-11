# Copyright 2025 Garena Online Private Limited
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

import random
import time

import launchpad as lp
from tqdm import tqdm

from oat.learners.base import LearnerBase


class OfflineLearner(LearnerBase):
    def run(self):
        self._init(self.args, self.actors)

        self.steps = 0
        self.start_time = time.time()

        self.actor_info = {}
        bs = self.args.rollout_batch_size_per_device

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True)

        self.steps = 1
        self.gradient_update_st = time.time()
        for p_ep in range(self.args.num_prompt_epoch):
            progress_bar = tqdm(
                range(len(self.all_buffer) // bs),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for ndx in range(0, len(self.all_buffer), bs):
                # Directly fetch from pre-loaded buffer instead of collecting preference data online.
                self.pi_buffer.extend(
                    self.all_buffer[ndx : min(ndx + bs, len(self.all_buffer))]
                )
                self.prompt_consumed += bs
                self.query_step += bs

                if self.steps % self.update_interval == 0:
                    self._pre_learning()
                    train_info = self.learn(self.steps // self.update_interval)
                    self._post_learning()

                    self.eval_and_log(train_info)

                progress_bar.update()
                self.steps += 1
            self.prompt_epoch = p_ep + 1
            # Reorder data for another epoch.
            random.Random(self.args.seed + p_ep).shuffle(self.all_buffer)

        self.eval_and_log(train_info, eval=True, save=True)

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            try:
                lp.stop()
            except AssertionError:
                pass
