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

import abc
import dataclasses
import math
import os
import socket
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Union
from warnings import warn

import deepspeed
import launchpad as lp
import Levenshtein
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
import vllm
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler

from oat.actor import Actor
from oat.args import OATArgs
from oat.model import LLM
from oat.types import DAPAlgo, PreferenceData
from oat.utils.data import PreferenceDataset, get_datasets, get_tokenizer
from oat.utils.deepspeed import get_strategy
from oat.utils.distributed import (
    init_process_group,
    node_ip_address_from_perspective,
    torch_type_codec,
)
from oat.utils.ipc import PlasmaShmClient, PlasmaShmServer
from oat.utils.launcher import DistributedLauncher


class LearnerBase(abc.ABC, DistributedLauncher):
    """Learner updates the LLM policy from preference data collected by actors."""

    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str,
        master_port: str,
        is_master: bool,
        args: OATArgs,
        actors: List[Actor],
        ipc_server: PlasmaShmServer,
    ) -> None:
        super().__init__(
            world_size, rank, local_rank, master_addr, master_port, is_master
        )
        self.args = args
        self.actors = actors
        self.ipc_server = ipc_server

    def _init(self, args: OATArgs, actors: List[Actor]) -> None:
        args, strategy = get_strategy(args)
        strategy.setup_distributed()

        model = LLM(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            ds_config=strategy.get_ds_train_config(is_wrapped=True),
        )
        self.algo = args.dap_algo

        if self.algo != DAPAlgo.SimPO:
            strategy.print("Running reference-based algorithm... (DPO, IPO, etc.)")
            assert args.ref_pretrain, "Reference model must be non-empty"
            ref_model = LLM(
                args.ref_pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
            )
        else:
            strategy.print("Running reference-free algorithm... (SimPO)")

        tokenizer = get_tokenizer(
            args.pretrain,
            model.model,
            "left",
            use_fast=not args.disable_fast_tokenizer,
        )

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant
                }
            )

        optimizer = strategy.create_optimizer(
            model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
        )

        # prepare datasets
        self.pi_buffer = deque(maxlen=args.pi_buffer_maxlen_per_device)
        self.all_buffer = deque(maxlen=int(1e9))

        self.prepare_data(strategy, tokenizer)
        strategy.print("Prompt dataset example:")
        strategy.print(self.prompts_dataset[0])
        strategy.print("Prompt dataset len:", len(self.prompts_dataset))

        self.eval_input_key = args.eval_input_key or args.input_key
        self.eval_output_key = args.eval_output_key or args.output_key

        # configure scheduler
        num_policy_sgd_steps_per_episodes = int(
            len(self.prompts_dataset) * args.max_epochs // args.train_batch_size
        )
        max_steps = math.ceil(
            args.num_prompt_epoch
            * num_policy_sgd_steps_per_episodes
            * args.max_step_adjustment
        )
        scheduler = get_scheduler(
            "cosine_with_min_lr",
            optimizer,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
        )
        strategy.print(
            f"num_policy_sgd_steps_per_episodes={num_policy_sgd_steps_per_episodes}; max_steps={max_steps}"
        )

        # prepare models/optimizers...
        if self.algo != DAPAlgo.SimPO:
            ((self.model, self.optimizer, self.scheduler), self.ref_model) = (
                strategy.prepare(
                    (model, optimizer, scheduler),
                    ref_model,
                    is_rlhf=True,
                )
            )
        else:
            (self.model, self.optimizer, self.scheduler) = strategy.prepare(
                (model, optimizer, scheduler),
                is_rlhf=True,
            )
            self.ref_model = None

        exp_name = args.wb_run_name + "_" + datetime.now().strftime("%m%dT%H:%M:%S")
        self.save_path = os.path.join(args.save_path, exp_name)
        os.makedirs(self.save_path, exist_ok=True)

        # logger
        self._wandb = None
        if strategy.args.use_wb and strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wb)
            wandb.init(
                entity=args.wb_org,
                project=args.wb_project,
                group=args.wb_group,
                name=exp_name,
                config=args.__dict__,
                reinit=True,
            )

        if actors:
            self.ipc_client = PlasmaShmClient(self.ipc_server)

        self.strategy = strategy
        self.tokenizer = tokenizer
        self.update_interval = args.rollout_batch_size // (
            strategy.world_size * args.rollout_batch_size_per_device
        )

        self.global_step = 0
        self.pi_beta_version = 0
        self.policy_sgd_step = 0
        self.query_step = 0
        self.prompt_consumed = 0
        self.prompt_epoch = 0
        self.gradient_update_elapse = np.nan

        # Log summary of the learner
        strategy.print(self.model)
        strategy.print(self.optimizer)
        strategy.print(self.scheduler)
        strategy.pprint(vars(args))
        strategy.print(f"Update interval = {self.update_interval}")

        # prepare parameter syncing to actors (reference to openrlhf)
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if actors and strategy.is_rank_0():
            master_addr = node_ip_address_from_perspective()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = len(actors) + 1
            backend = "nccl"
            if vllm.__version__ > "0.4.2":
                backend = "gloo"
                warn(f"Using gloo backend for vLLM version {vllm.__version__}")
            futs = [
                actor.futures.init_process_group(
                    master_addr,
                    master_port,
                    i + 1,
                    world_size,
                    "oat",
                    backend=backend,
                )
                for i, actor in enumerate(actors)
            ]
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="oat",
            )
            _ = [fut.result() for fut in futs]

        dist.barrier()

    def prepare_data(self, strategy, tokenizer):
        self.prompts_dataset, self.eval_prompts_dataset = get_datasets(
            tokenizer, strategy
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
        )
        self.eval_prompts_dataloader = DataLoader(
            self.eval_prompts_dataset,
            batch_size=strategy.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def collect_preference(
        self,
        prompts: Union[str, List[str]],
        formatted_prompts: List[str],
        refs: Union[str, List[str]],
    ):
        # generate response & get feedback
        st_time = time.time()
        rank = torch.distributed.get_rank()
        actor = self.actors[rank % len(self.actors)]

        if self.strategy.args.online_evaluation:
            handle = actor.step(prompts, formatted_prompts, refs)
        else:
            handle = actor.step(prompts, formatted_prompts)

        preference_data: List[PreferenceData] = self.ipc_client.deserialize_ipc(handle)

        actor_time = time.time() - st_time

        metric = {
            "actor/total_time": actor_time,
            "actor/chosen_avg_str_len": np.mean(
                [len(p.chosen_response) for p in preference_data]
            ),
            "actor/rejected_avg_str_len": np.mean(
                [len(p.rejected_response) for p in preference_data]
            ),
            "actor/init_clash_ratio": np.mean([p.init_clash for p in preference_data]),
            "actor/same_response_ratio": np.mean([p.same for p in preference_data]),
            "actor/pair_edit_dist": np.mean(
                [
                    Levenshtein.distance(p.chosen_response, p.rejected_response)
                    for p in preference_data
                ]
            ),
            "actor/chosen_id": np.mean([p.chosen_id for p in preference_data]),
        }

        mean_info = tree.map_structure(
            lambda *x: np.mean(x), *[p.info for p in preference_data]
        )
        metric.update(mean_info)

        return preference_data, metric

    def run(self):
        self._init(self.args, self.actors)

        self.steps = 0
        early_stop = False
        self.start_time = time.time()

        self.actor_info = {}

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True)

        self.steps = 1
        self.gradient_update_st = time.time()
        for p_ep in range(self.args.num_prompt_epoch):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(p_ep)
                self.strategy.print(f"Set DistributedSampler at epoch {p_ep}")
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
            )

            for processed_prompts, raw_prompts, refs in self.prompts_dataloader:
                if early_stop:
                    break
                preference_data, self.actor_info = self.collect_preference(
                    raw_prompts, processed_prompts, refs
                )
                self.prompt_consumed += len(processed_prompts)
                self.query_step += np.sum(
                    [not p.is_model_data for p in preference_data]
                )
                self.process_preference_data(preference_data, raw_prompts)

                if self.steps % self.update_interval == 0:
                    train_info = self.preference_learning(
                        self.steps // self.update_interval
                    )

                    self.eval_and_log(train_info)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        self.sync_params_to_actors()

                    if (
                        self.steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                progress_bar.update()
                self.steps += 1

                if self.get_current_query() > self.args.max_queries:
                    early_stop = True

            self.prompt_epoch = p_ep + 1

        self.eval_and_log(train_info, eval=True)

        if self.args.dump_all_buffer:  # For debug purpose.
            if not self.strategy.is_rank_0():
                dist.gather_object(self.all_buffer)
            else:
                gather_all_buffer = [None] * self.strategy.world_size
                dist.gather_object(self.all_buffer, gather_all_buffer)
                pd.to_pickle(
                    gather_all_buffer, os.path.join(self.save_path, "all_buffer.pkl")
                )

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            lp.stop()

    def process_preference_data(self, data_list: List[PreferenceData], raw_prompts):
        for i, pref in enumerate(data_list):
            # Replace with raw prompts instead of templated ones.
            new_pref = dataclasses.replace(pref, prompt=raw_prompts[i])  # shallow copy
            self.pi_buffer.append(new_pref)
            if self.args.dump_all_buffer:
                c = new_pref.chosen_response
                r = new_pref.rejected_response
                self.all_buffer.append(
                    PreferenceData(
                        prompt=new_pref.prompt,
                        chosen_response=c,
                        rejected_response=r,
                        same=c == r,
                    )
                )

    def preference_learning(self, learning_round):
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = PreferenceDataset(
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
            acc_mean = []
            loss_mean = []
            reward_margin = []
            learn_batch_time = []
            self.model.train()
            st = time.time()
            for data in dataloader:
                if local_sgd_steps > self.args.max_sgd_steps:
                    break
                infos = self.learning_step(data)

                # metrics
                loss = infos.pop("loss")
                chosen_reward = infos.pop("chosen_reward")
                rejected_reward = infos.pop("rejected_reward")
                acc_mean.append((chosen_reward > rejected_reward).float().mean().item())
                loss_mean.append(loss.cpu().item())
                reward_margin.append((chosen_reward - rejected_reward).mean().item())

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
            "chosen_reward": chosen_reward.mean().item(),
            "rejected_reward": rejected_reward.mean().item(),
            "acc_mean": np.mean(acc_mean),
            "loss_mean": np.mean(loss_mean),
            "reward_margin": np.mean(reward_margin),
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

    @abc.abstractmethod
    def learning_step(self, data):
        """Preference learning step."""

    def get_misc_info(self) -> Dict[str, Any]:
        return {
            "pi_beta_version": self.pi_beta_version,
            "global_step": self.global_step,
            "policy_sgd_step": self.policy_sgd_step,
            "pi_buffer_len": len(self.pi_buffer),
            "elapse": time.time() - self.start_time,
            "update_interval": self.update_interval,
            "prompt_epoch": self.prompt_epoch,
            "gradient_update_elapse": self.gradient_update_elapse,
        }

    def get_current_query(self):
        return self.strategy.all_reduce(self.query_step, op="sum")

    def _should_eval(self):
        if not hasattr(self, "_pending_eval"):
            self._pending_eval = False

        do_eval = self.steps % self.args.eval_steps == 0
        if not (do_eval or self._pending_eval):
            return False
        else:
            if do_eval and not hasattr(self, "last_eval_query_step"):
                self.last_eval_query_step = self.get_current_query()
                return True
            query_step_elapse = self.get_current_query() - self.last_eval_query_step
            if query_step_elapse < self.args.eval_query_interval:
                self._pending_eval = True
                return False
            self._pending_eval = False
            self.last_eval_query_step = self.get_current_query()
            return True

    def eval_and_log(self, train_info, eval=False):
        # eval
        eval_info = {}
        if eval or self._should_eval():
            eval_info = self.evaluate(self.eval_prompts_dataloader, self.steps)

        # logs
        if eval_info or self.steps % self.args.logging_steps == 0:
            misc_info = self.get_misc_info()
            last_lr = self.scheduler.get_last_lr()[0]
            misc_info["lr"] = last_lr

            misc_info = {
                "misc/%s" % k: v
                for k, v in {
                    **misc_info,
                }.items()
            }
            logs_dict = {**train_info, **eval_info, **self.actor_info, **misc_info}
            logs_dict = self.strategy.all_reduce(logs_dict)
            logs_dict.update(
                self.strategy.all_reduce(
                    {
                        "misc/query_step": self.query_step,
                        "misc/prompt_consumed": self.prompt_consumed,
                    },
                    op="sum",
                )
            )

            if self.strategy.is_rank_0():
                if self.pi_buffer:
                    self.strategy.pprint(np.random.choice(self.pi_buffer))
                self.strategy.pprint(logs_dict)
                if self._wandb is not None:
                    self._wandb.log(logs_dict)

    def evaluate(self, dataloader, steps):
        self.strategy.print(f"Start generating evaluation responses at step {steps}")
        st_time = time.time()
        # 1) Let Actors cache the current behavior policy.
        if self.strategy.is_rank_0():
            done = [actor.futures.notify_eval_start() for actor in self.actors]
            _ = [d.result() for d in done]

        # 2) Push the latest policy for fast vLLM generation.
        dist.barrier()
        self._broadcast_to_vllm()

        # 3) Generate and process results

        win_rate = 0
        win_rate_prob = 0
        if self.strategy.is_rank_0():
            processed_prompts = []
            prompts = []
            responses = []
            references = []
            futs = []
            win_probs = []
            wins = []
            progress_bar = tqdm(range(len(dataloader)), desc="Evaluating")
            for i, (batch_processed_prompts, batch_prompts, refs) in enumerate(
                dataloader
            ):
                processed_prompts.extend(batch_processed_prompts)
                prompts.extend(batch_prompts)
                references.extend(refs)

                actor = self.actors[i % len(self.actors)]
                fut = actor.futures.generate_and_maybe_eval(
                    batch_prompts, batch_processed_prompts, refs
                )
                futs.append(fut)
                if len(futs) == len(self.actors) or i == len(dataloader) - 1:
                    for fut in futs:
                        resp, win_prob = fut.result()
                        responses.extend(resp)
                        wins.extend(win_prob > 0.5)
                        win_probs.extend(win_prob)
                    futs.clear()
                progress_bar.update()

            eval_res_path = os.path.join(self.save_path, "eval_results")
            os.makedirs(eval_res_path, exist_ok=True)
            pd.DataFrame(
                {
                    self.eval_input_key: prompts,
                    self.eval_output_key: responses,
                    f"format_{self.eval_input_key}": processed_prompts,
                    "reference": references,
                    "generator": self.args.wb_run_name,
                }
            ).to_json(
                os.path.join(eval_res_path, f"{steps}.json"),
                orient="records",
            )
            win_rate = np.mean(wins).item()
            win_rate_prob = np.mean(win_probs).item()

        win_rate = self.strategy.broadcast(win_rate)
        win_rate_prob = self.strategy.broadcast(win_rate_prob)

        # 4) Recover Actors' original behavior policy.
        if self.strategy.is_rank_0():
            done = [actor.futures.notify_eval_done() for actor in self.actors]
            _ = [d.result() for d in done]

        return {
            "eval/rm_win_rate": win_rate,
            "eval/rm_win_rate_prob": win_rate_prob,
            "eval/elapse": time.time() - st_time,
        }

    def sync_params_to_actors(self):
        self._broadcast_to_vllm()
        self.pi_beta_version += 1

    def _broadcast_to_vllm(self):
        model = self.model.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # Fire all vllm engines for broadcast
            if self.strategy.is_rank_0():
                shape = (
                    param.shape
                    if self.strategy.args.zero_stage != 3
                    else param.ds_shape
                )
                futs = [
                    actor.futures.update_weight(
                        name,
                        dtype=torch_type_codec(param.dtype),
                        shape=shape,
                        empty_cache=count == num_params,
                    )
                    for actor in self.actors
                ]

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters(
                [param], enabled=self.strategy.args.zero_stage == 3
            ):
                if self.strategy.is_rank_0():
                    dist.broadcast(param.data, 0, group=self._model_update_group)
                    _ = [fut.result() for fut in futs]
