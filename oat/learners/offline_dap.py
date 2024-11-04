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

import os
import random
import time
from typing import List

import launchpad as lp
import pandas as pd
import torch
import torch.distributed as dist
import tree
from torch.utils.data import DataLoader
from tqdm import tqdm

from oat.learners.dap import DAPLearner
from oat.types import PreferenceData
from oat.utils.data import get_datasets, load_data_from_disk_or_hf, shard_buffer


class OfflineDAPLearner(DAPLearner):

    def prepare_data(self, strategy, tokenizer):
        """Load offline preference data into the buffer instead of using online generated data."""
        args = self.args
        if args.preference_data:
            data = load_data_from_disk_or_hf(args.preference_data)
            all_shards = []
            for item in tqdm(data, desc="loading preference data"):
                all_shards.append(
                    PreferenceData(
                        prompt=item[args.prompt_key],
                        chosen_response=item[args.chosen_key],
                        rejected_response=item[args.rejected_key],
                        chosen_id=0,
                        chosen_feature=None,
                        rejected_feature=None,
                        init_clash=False,
                        same=item[args.chosen_key] == item[args.rejected_key],
                        is_model_data=False,
                        info={},
                    )
                )
            all_shards = all_shards[: args.max_train]
            self.all_buffer: List[PreferenceData] = shard_buffer(
                all_shards,
                dist.get_rank(),
                dist.get_world_size(),
                args.seed,
                shuffle=True,
                drop_last=True,
            )
        else:
            # Load pre-dumped data.
            assert os.path.exists(args.offline_buffer_path)
            all_shards = pd.read_pickle(args.offline_buffer_path)
            self.all_buffer: List[PreferenceData] = list(
                all_shards[torch.distributed.get_rank()]
            )
        self.prompts_dataset = tree.flatten(
            all_shards
        )  # needed to calculate lr scheduler
        self.prompts_dataloader = None
        _, self.eval_prompts_dataset = get_datasets(tokenizer, strategy, eval_only=True)
        self.eval_prompts_dataloader = DataLoader(
            self.eval_prompts_dataset,
            batch_size=strategy.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def run(self):
        self._init(self.args, self.actors)

        self.steps = 0
        self.start_time = time.time()

        self.actor_info = {}
        bs = self.args.rollout_batch_size_per_device

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True)

        self.steps = 1
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
                    train_info = self.preference_learning(
                        self.steps // self.update_interval
                    )

                    self.eval_and_log(train_info)

                progress_bar.update()
                self.steps += 1
            self.prompt_epoch = p_ep + 1
            # Reorder data for another epoch.
            random.Random(self.args.seed + p_ep).shuffle(self.all_buffer)

        self.eval_and_log(train_info, eval=True)

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            lp.stop()
