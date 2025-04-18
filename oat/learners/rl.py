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

"""RL optimizer."""

import math
import time
from typing import List

import numpy as np
import torch
import tree
from torch import distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.learners.base import LearnerBase
from oat.model import LLM, Critic
from oat.types import RLAlgo, TrajectoryData
from oat.utils.data import TrajectoryDataset
from oat.utils.ops import disable_dropout


class RLLearner(LearnerBase):
    """Policy learning through RL algorithms."""

    def _init(self, args: OATArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        assert self.algo in RLAlgo

        if self.algo == RLAlgo.PPO and (args.beta > 0 or args.kl_penalty_coef > 0):
            # Reference policy for regularization.
            self.strategy.print("Running KL-regularized algorithm...")
            assert args.ref_pretrain, "Reference model must be non-empty"
            self.ref_model = LLM(
                args.ref_pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
            )
            disable_dropout(self.ref_model)

            # prepare models/optimizers...
            ((self.model, self.optimizer, self.scheduler), self.ref_model) = (
                self.strategy.prepare(
                    (self.model, self.optimizer, self.scheduler),
                    self.ref_model,
                )
            )
        else:
            self.strategy.print("Running reference-free algorithm...")
            (self.model, self.optimizer, self.scheduler) = self.strategy.prepare(
                (self.model, self.optimizer, self.scheduler),
            )
            self.ref_model = None

        if args.critic_type == "ppo":
            self.strategy.print("Learning critic online...")
            self.critic = Critic(
                args.critic_pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules,
                ds_config=self.strategy.get_ds_train_config(is_wrapped=True),
            )
            disable_dropout(self.critic)
            if args.gradient_checkpointing:
                self.critic.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": args.gradient_checkpointing_use_reentrant
                    }
                )
            self.critic_optimizer = self.strategy.create_optimizer(
                self.critic,
                lr=args.critic_learning_rate,
                betas=(args.adam_beta_1, args.adam_beta_2),
                weight_decay=args.l2,
            )
            max_steps_to_schedule = self.max_steps * args.critic_max_step_adjustment
            scheduler_specific_kwargs = {}
            if args.lr_scheduler not in ["polynomial"]:
                scheduler_specific_kwargs["min_lr"] = args.learning_rate * 0.1
            self.critic_scheduler = get_scheduler(
                args.lr_scheduler,
                self.critic_optimizer,
                num_warmup_steps=math.ceil(
                    max_steps_to_schedule * args.lr_warmup_ratio
                ),
                num_training_steps=max_steps_to_schedule,
                scheduler_specific_kwargs=scheduler_specific_kwargs,
            )

            (self.critic, self.critic_optimizer, self.critic_scheduler) = (
                self.strategy.prepare(
                    (self.critic, self.critic_optimizer, self.critic_scheduler),
                )
            )
        else:
            self.critic = None

        self.dataset_builder = TrajectoryDataset
        dist.barrier()

    def process_feedback_data(self, data_list: List[TrajectoryData]):
        self.query_step += len(data_list)
        for trajectory in data_list:
            self.pi_buffer.append(trajectory)
            if self.args.dump_all_buffer:
                self.all_buffer.append(
                    TrajectoryData(
                        prompt=trajectory.prompt,
                        response=trajectory.response,
                        rewards=trajectory.rewards,
                    )
                )

    def learn(self, learning_round: int):
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.args.prompt_max_length,
            self.args.generate_max_length,
            self.strategy,
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
            learn_batch_time = []

            self.model.train()
            st = time.time()
            for data in dataloader:
                if local_sgd_steps > self.args.max_sgd_steps:
                    break
                infos = self.learning_step(data)

                step_bar.update()
                self.global_step += 1
                if self.global_step % self.strategy.accumulated_gradient == 0:
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
