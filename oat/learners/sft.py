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

"""SFT optimizer for imitation learning."""

import logging
import time
from contextlib import nullcontext
from typing import List, Tuple

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import tree
from torch.autograd.graph import save_on_cpu
from torch.utils.data import DataLoader
from tqdm import tqdm

from oat.learners.offline import OfflineLearner
from oat.types import TrajectoryData
from oat.utils.data import (
    TrajectoryDataset,
    get_datasets,
    load_data_from_disk_or_hf,
    shard_buffer,
)
from oat.utils.lm_head import FusedLinear


class SFTLearner(OfflineLearner):
    """Policy learning via supervised learning."""

    def _init(self, args, actors):
        super()._init(args, actors)
        (self.model, self.optimizer, self.scheduler) = self.strategy.prepare(
            (self.model, self.optimizer, self.scheduler),
        )
        self.ref_model = None
        self.dataset_builder = TrajectoryDataset
        dist.barrier()

    def prepare_data(self, strategy, tokenizer):
        """Load offline SFT chat data into the buffer instead of using online generated data."""
        args = self.args
        assert args.chat_data
        data = load_data_from_disk_or_hf(args.chat_data)[args.train_split]
        all_shards = []
        drop_cnt = 0
        for item in tqdm(
            data, desc="loading SFT chat data", disable=not strategy.is_rank_0()
        ):
            format_prompt = tokenizer.apply_chat_template(
                item[args.msg_key],
                tokenize=True,
            )
            if len(format_prompt) >= args.max_model_len:
                drop_cnt += 1
                continue  # drop too long prompts
            all_shards.append(
                TrajectoryData(
                    trajectory_ids=None,
                    num_turns=None,
                    response_token_ranges=None,
                    turn_weights=None,
                    messages=item[args.msg_key],
                )
            )
        logging.info(f"[Dataset] Dropped {drop_cnt} samples with too long prompts.")

        all_shards = all_shards[: args.max_train]
        self.all_buffer: List[TrajectoryData] = shard_buffer(
            all_shards,
            dist.get_rank(),
            dist.get_world_size(),
            args.seed,
            shuffle=True,
            drop_last=True,
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

    def process_feedback_data(self, data_list):
        """Dummy function because SFT does not need this."""
        del data_list
        raise NotImplementedError

    def learn(self, learning_round):
        dataset = self.dataset_builder(
            buffer=self.pi_buffer,
            tokenizer=self.tokenizer,
            strategy=self.strategy,
        )
        if learning_round == 1:
            self.strategy.print("Training example")
            self.strategy.print(dataset[0])
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.train_batch_size_per_device,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        for epoch in range(self.args.max_epochs):
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )
            loss_mean = []
            learn_batch_time = []

            self.model.train()
            st = time.time()
            for data in dataloader:
                if local_sgd_steps > self.args.max_sgd_steps:
                    break
                infos = self.learning_step(data)

                # metrics
                loss = infos.pop("loss")
                loss_mean.append(loss.cpu().item())

                step_bar.update()
                self.global_step += 1
                if self.global_step % self.strategy.grad_acc_step == 0:
                    learn_batch_time.append(time.time() - st)
                    self.gradient_update_elapse = time.time() - self.gradient_update_st
                    st = time.time()
                    self.gradient_update_st = time.time()
                    self.policy_sgd_step += 1
                    local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "epoch": epoch + 1,
            "loss_mean": np.mean(loss_mean),
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        return train_info

    def learning_step(self, data):
        device = torch.cuda.current_device()
        input_ids, attention_mask, turn_weights, response_token_ranges = data
        del turn_weights  # Not using the weights for now.
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.squeeze(1).to(device)
        context = save_on_cpu() if self.args.activation_offloading else nullcontext()
        with context:
            loss = self.model_forward(
                self.model, input_ids, attention_mask, response_token_ranges
            )
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
        infos = {
            "loss": loss.detach(),
        }
        return infos

    def model_forward(self, model, input_ids, attention_mask, response_token_ranges):
        bs1 = True
        if len(input_ids) == 1:
            logits_to_keep = []
            for st, end in response_token_ranges[0]:
                logits_to_keep.append(torch.arange(st - 1, end - 1))

            logits_to_keep = torch.cat(logits_to_keep)
            logits_indices = slice(None, None)
            label_indices = logits_to_keep + 1
            sliced_mask = attention_mask[:, label_indices]
        else:
            bs1 = False
            logits_to_keep = 0
            logits_indices = slice(None, -1)
            label_indices = slice(1, None)
            sliced_mask = attention_mask.clone().bool()
            for mask, ranges in zip(sliced_mask, response_token_ranges):
                cursor = 0
                for st_idx, end_idx in ranges:
                    mask[cursor:st_idx] = False
                    cursor = end_idx
            sliced_mask = sliced_mask[:, 1:]

        model_output = model(
            input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            without_logits=self.args.use_fused_lm_head,
        )
        if self.args.use_fused_lm_head:
            # annoying but need to handle different naming
            hidden_states = model_output.last_hidden_state[
                :, logits_to_keep if bs1 else logits_indices
            ]
            vocab_weights = model.model.lm_head.weight

            fused_linear = FusedLinear(compute_entropy=False)
            with deepspeed.zero.GatheredParameters(
                [vocab_weights],
                fwd_module=fused_linear,
                enabled=self.strategy.args.zero_stage == 3,
            ):
                target_logps, _ent = fused_linear(
                    hidden_states,
                    vocab_weights,
                    input_ids[:, label_indices],
                    temperature=1,
                )
            del _ent
            sum_loss = (target_logps * sliced_mask).sum(-1)
            if not self.args.sft_sum_loss:
                sum_loss /= sliced_mask.sum(-1)
            sft_loss = -sum_loss.mean()  # average across examples
        else:
            batch_logps = self.get_batch_logps(
                model_output["logits"][:, logits_indices],
                input_ids[:, label_indices],
                sliced_mask,
                average_log_prob=not self.args.sft_sum_loss,
            )
            sft_loss = -batch_logps.mean()  # average across examples
        return sft_loss

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        completion_masks: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> Tuple[torch.Tensor]:
        assert logits.shape[:-1] == labels.shape
        labels = labels.clone()

        labels[completion_masks == False] = 0

        all_logp = logits.log_softmax(-1)
        target_logps = torch.gather(all_logp, dim=2, index=labels.unsqueeze(2)).squeeze(
            2
        )
        response_length = completion_masks.sum(-1)
        if average_log_prob:
            return (target_logps * completion_masks).sum(-1) / response_length
        else:
            return (target_logps * completion_masks).sum(-1)
