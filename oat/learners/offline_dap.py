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
import os
from typing import List

import pandas as pd
import torch
import torch.distributed as dist
import tree
from torch.utils.data import DataLoader
from tqdm import tqdm

from oat.learners.dap import DAPLearner
from oat.learners.offline import OfflineLearner
from oat.types import PreferenceData
from oat.utils.data import (
    extract_assistant_content,
    get_datasets,
    load_data_from_disk_or_hf,
    shard_buffer,
)


class OfflineDAPLearner(OfflineLearner, DAPLearner):

    def prepare_data(self, strategy, tokenizer):
        """Load offline preference data into the buffer instead of using online generated data."""
        args = self.args
        if args.preference_data:
            data = load_data_from_disk_or_hf(args.preference_data)[args.train_split]
            all_shards = []
            drop_cnt = 0
            for item in tqdm(
                data, desc="loading preference data", disable=not strategy.is_rank_0()
            ):
                if args.apply_chat_template and tokenizer.chat_template:
                    format_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": item[args.prompt_key]}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                else:
                    format_prompt = tokenizer.encode(item[args.prompt_key])
                if len(format_prompt) >= args.prompt_max_length:
                    drop_cnt += 1
                    continue  # drop too long prompts
                chosen = item[args.chosen_key]
                reject = item[args.rejected_key]
                if args.extract_content:
                    chosen = extract_assistant_content(chosen)
                    reject = extract_assistant_content(reject)
                all_shards.append(
                    PreferenceData(
                        prompt=item[args.prompt_key],
                        chosen_response=chosen,
                        rejected_response=reject,
                        chosen_id=0,
                        chosen_feature=None,
                        rejected_feature=None,
                        init_clash=False,
                        loss_mask=True,
                        is_model_data=False,
                        info={},
                    )
                )
            logging.info(f"[Dataset] Dropped {drop_cnt} samples with too long prompts.")

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
        if args.eval_steps > 0:
            _, self.eval_prompts_dataset = get_datasets(
                tokenizer, strategy, eval_only=True
            )
            self.eval_prompts_dataloader = DataLoader(
                self.eval_prompts_dataset,
                batch_size=strategy.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
