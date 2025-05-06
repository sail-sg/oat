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
import logging
import math
import os
import socket
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import deepspeed
import launchpad as lp
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.collectors import AsyncFeedbackCollector, FeedbackCollector
from oat.model import LLM
from oat.types import PreferenceData, TrajectoryData
from oat.utils.data import get_datasets, get_tokenizer
from oat.utils.deepspeed import get_strategy
from oat.utils.distributed import (
    init_process_group,
    node_ip_address_from_perspective,
    torch_type_codec,
)
from oat.utils.ipc import PlasmaShmClient, PlasmaShmServer
from oat.utils.launcher import DistributedLauncher
from oat.utils.ops import disable_dropout


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
        actors: List[ActorBase],
        ipc_server: PlasmaShmServer,
    ) -> None:
        super().__init__(
            world_size, rank, local_rank, master_addr, master_port, is_master
        )
        self.args = args
        self.actors = actors
        self.ipc_server = ipc_server

    def _init(self, args: OATArgs, actors: List[ActorBase]) -> None:
        args, strategy = get_strategy(args)
        strategy.setup_distributed()

        # Init actors async.
        actor_init_futs = None
        if actors and strategy.is_rank_0():
            actor_init_futs = [actor.futures.init(i) for i, actor in enumerate(actors)]

        # ---------- Model related ----------
        # init policy model
        self.model = LLM(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            # ds_config=strategy.get_ds_train_config(is_wrapped=True),
        )
        disable_dropout(self.model)
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant
                }
            )
        # load tokenizer
        tokenizer_path = args.tokenizer if args.tokenizer else args.pretrain
        strategy.print("Loading tokenizer from:", tokenizer_path)

        self.tokenizer = get_tokenizer(
            tokenizer_path,
            self.model.model,
            "left",
            use_fast=not args.disable_fast_tokenizer,
        )
        strategy.print("chat template:", self.tokenizer.chat_template)

        # ---------- Data related ----------
        # prepare buffer
        self.pi_buffer = deque(maxlen=args.pi_buffer_maxlen_per_device)
        self.all_buffer = deque(maxlen=int(1e9))
        # prepare (eval) prompts dataloader
        self.prepare_data(strategy, self.tokenizer)

        strategy.print("Prompt dataset example:")
        strategy.print(self.prompts_dataset[0])
        strategy.print("Prompt dataset len:", len(self.prompts_dataset))

        self.eval_input_key = args.eval_input_key or args.input_key
        self.eval_output_key = args.eval_output_key or args.output_key

        # ---------- Optimizer related ----------
        self.optimizer = strategy.create_optimizer(
            self.model,
            lr=args.learning_rate,
            betas=(args.adam_beta_1, args.adam_beta_2),
            weight_decay=args.l2,
        )
        num_policy_sgd_steps_per_episodes = int(
            len(self.prompts_dataset) * args.max_epochs // args.train_batch_size
        )
        self.max_steps = math.ceil(
            args.num_prompt_epoch * num_policy_sgd_steps_per_episodes
        )
        max_steps_to_schedule = self.max_steps * args.max_step_adjustment

        scheduler_specific_kwargs = {}
        if args.lr_scheduler in ["cosine_with_min_lr"]:
            scheduler_specific_kwargs["min_lr"] = args.learning_rate * 0.1
        self.scheduler = get_scheduler(
            args.lr_scheduler,
            self.optimizer,
            num_warmup_steps=math.ceil(max_steps_to_schedule * args.lr_warmup_ratio),
            num_training_steps=max_steps_to_schedule,
            scheduler_specific_kwargs=scheduler_specific_kwargs,
        )
        strategy.print(
            f"num_policy_sgd_steps_per_episodes={num_policy_sgd_steps_per_episodes}; max_steps={max_steps_to_schedule}"
        )

        # prepare collector, which communicates with actors
        if actors:
            if self.args.asynchronous:
                self.collector = AsyncFeedbackCollector(
                    args, actors, PlasmaShmClient(self.ipc_server)
                )
            else:
                self.collector = FeedbackCollector(
                    args, actors, PlasmaShmClient(self.ipc_server)
                )
        else:
            strategy.print("No actors or feedback collector in offline mode.")

        exp_name = args.wb_run_name + "_" + datetime.now().strftime("%m%dT%H:%M:%S")
        self.save_path = os.path.join(args.save_path, exp_name)
        if strategy.is_rank_0():
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

        self.algo = args.algo
        self.strategy = strategy
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
        self.weight_sync_elapse = np.nan
        self.vllm_wake_up_time = 0
        self.vllm_go_sleep_time = 0
        self.pi_beta_lags_behind = False

        # Log summary of the learner
        strategy.print(self.model)
        strategy.print(self.optimizer)
        strategy.print(self.scheduler)
        strategy.pprint(vars(args))
        strategy.print(f"Update interval = {self.update_interval}")

        if actor_init_futs is not None:
            _ = [fut.result() for fut in actor_init_futs]

        # prepare parameter syncing to actors (reference to openrlhf)
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        backend = "gloo" if self.args.collocate else "nccl"
        if actors and strategy.is_rank_0():
            master_addr = node_ip_address_from_perspective()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = len(actors) + 1
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

    def run(self):
        """Entry point of the entire program."""
        self._init(self.args, self.actors)

        if self.args.resume_dir:
            # Resume from previous training.
            # 1) Model & training states
            self.strategy.load_ckpt(
                self.model.model, self.args.resume_dir, self.args.resume_tag
            )
            # 2) Dataset ... (TODO)

        self.steps = 0
        early_stop = False
        self.start_time = time.time()

        self.actor_info = {}

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True, save=False)

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
                # Call actor.step remotely to generate rollout & collect feedback.
                feedback_data, self.actor_info = self.collector.collect_feedback(
                    raw_prompts, processed_prompts, refs
                )
                dist.barrier()

                if feedback_data is None:
                    # Asynchronous prefilling, data is stored in collector's buffer.
                    continue
                self.prompt_consumed += len(feedback_data)

                self.process_feedback_data(feedback_data)

                if (
                    self.args.dump_replay_every > 0
                    and self.steps % self.args.dump_replay_every == 0
                ):
                    if not self.strategy.is_rank_0():
                        dist.gather_object(self.pi_buffer)
                    else:
                        gather_all_buffer = [None] * self.strategy.world_size
                        dist.gather_object(self.pi_buffer, gather_all_buffer)
                        pd.to_pickle(
                            (processed_prompts, refs, gather_all_buffer),
                            os.path.join(
                                self.save_path, f"buffer_step{self.steps:05}.pkl"
                            ),
                        )

                if self.steps % self.update_interval == 0:
                    self._pre_learning()
                    train_info = self.learn(self.steps // self.update_interval)
                    self._post_learning()

                    if (
                        self.steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        self.sync_params_to_actors()

                    if (
                        self.steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                    self.eval_and_log(train_info)

                progress_bar.update()
                self.steps += 1

                if self.get_current_query() > self.args.max_queries:
                    early_stop = True

            self.prompt_epoch = p_ep + 1

        self.eval_and_log(train_info, eval=True, save=True)

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

    @abc.abstractmethod
    def process_feedback_data(
        self, data_list: List[Union[PreferenceData, TrajectoryData]]
    ):
        """Process collected feedback data, e.g., adding it to buffer."""

    @abc.abstractmethod
    def learn(self, learning_round: int):
        """Agent learning given the current data in the buffer."""

    @abc.abstractmethod
    def learning_step(self, data):
        """Agent learning step."""

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
        prompt_id_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)

        Returns:
            all_logp: all log prob of shape (batch_size, sequence_length, vocab_size)
            target_logps: target log prob of shape (batch_size, sequence_length)
            completion_masks: mask=True if it is completion's token, shape (batch_size, sequence_length)
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        completion_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(completion_masks, prompt_id_lens):
            mask[:source_len] = False
        completion_masks = completion_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[completion_masks == False] = 0

        all_logp = logits.log_softmax(-1)
        target_logps = torch.gather(all_logp, dim=2, index=labels.unsqueeze(2)).squeeze(
            2
        )

        return all_logp, target_logps, completion_masks

    def get_misc_info(self) -> Dict[str, Any]:
        return {
            "pi_beta_version": self.pi_beta_version,
            "global_step": self.global_step,
            "policy_sgd_step": self.policy_sgd_step,
            "pi_buffer_len": len(self.pi_buffer),
            "prompt_dataset_len": len(self.prompts_dataset),
            "elapse": time.time() - self.start_time,
            "update_interval": self.update_interval,
            "prompt_epoch": self.prompt_epoch,
            "gradient_update_elapse": self.gradient_update_elapse,
            "weight_sync_elapse": self.weight_sync_elapse,
            "vllm_go_sleep_time": self.vllm_go_sleep_time,
            "vllm_wake_up_time": self.vllm_wake_up_time,
        }

    def get_current_query(self):
        return self.strategy.all_reduce(self.query_step, op="sum")

    def _should_do(self, interval_steps):
        if interval_steps <= 0:
            return False
        if not hasattr(self, "_pending_eval"):
            self._pending_eval = False

        do_eval = self.steps % interval_steps == 0
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

    def eval_and_log(self, train_info, eval=False, save=False):
        # eval
        eval_info = {}
        if (self.args.eval_steps > 0 and eval) or self._should_do(self.args.eval_steps):
            eval_info = self.evaluate(self.eval_prompts_dataloader, self.steps)

        # save
        if (self.args.save_steps > 0 and save) or (
            self.steps > 0
            and self._should_do(self.args.save_steps)
            and self.steps >= self.args.save_from
        ):
            self.strategy.save_model(
                self.model,
                self.tokenizer,
                os.path.join(self.save_path, "saved_models"),
                tag="step_{:05d}".format(self.steps),
                max_num=self.args.max_save_num,
                max_mem=self.args.max_save_mem,
            )
            if self.args.save_ckpt:
                self.strategy.save_ckpt(
                    self.model.model,
                    os.path.join(self.save_path, "checkpoints"),
                    tag="step_{:05d}".format(self.steps),
                    max_num=self.args.max_save_num,
                    max_mem=self.args.max_save_mem,
                )

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
                    self.strategy.print(np.random.choice(self.pi_buffer))
                self.strategy.pprint(logs_dict)
                if self._wandb is not None:
                    self._wandb.log(logs_dict)

    def _pre_evaluate(self):
        # Let Actors cache the current behavior policy.
        if self.strategy.is_rank_0():
            done = [
                actor.futures.notify_eval_start(self.pi_beta_lags_behind)
                for actor in self.actors
            ]
            _ = [d.result() for d in done]
        dist.barrier()

        # Sync the latest policy to vLLM engines.
        if self.pi_beta_lags_behind:
            self._broadcast_to_vllm()

    def _post_evaluate(self):
        # Recover Actors' original behavior policy.
        if self.strategy.is_rank_0():
            done = [
                actor.futures.notify_eval_done(self.pi_beta_lags_behind)
                for actor in self.actors
            ]
            _ = [d.result() for d in done]
        dist.barrier()

    def evaluate(self, dataloader, steps):
        self.strategy.print(f"Start generating evaluation responses at step {steps}")
        st_time = time.time()

        assert not self.pi_beta_lags_behind, "pi beta lags behind for evaluation"
        self._pre_evaluate()

        # Generate and process results
        win_rate = 0
        scores = 0
        accuracy = 0
        response_len = 0
        eval_count = 0
        if self.strategy.is_rank_0():
            processed_prompts = []
            prompts = []
            responses = []
            references = []
            futs = []
            scores = []
            wins = []
            accuracies = []
            progress_bar = tqdm(range(len(dataloader)), desc="Evaluating")
            for i, (batch_processed_prompts, batch_prompts, refs) in enumerate(
                dataloader
            ):
                eval_count += len(batch_prompts)
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
                        resp, score = fut.result()
                        responses.extend(resp)
                        wins.extend(score > 0.5)  # For preference learning.
                        accuracies.extend(score == 1)  # For RL with verifiable rewards.
                        scores.extend(score)
                    futs.clear()
                progress_bar.update()

            eval_res_path = os.path.join(self.save_path, "eval_results")
            os.makedirs(eval_res_path, exist_ok=True)
            pd.DataFrame(
                {
                    self.eval_input_key: prompts,
                    "output": responses,
                    "scores": scores,
                    f"format_{self.eval_input_key}": processed_prompts,
                    "reference": references,
                    "generator": self.args.wb_run_name,
                }
            ).to_json(
                os.path.join(eval_res_path, f"{steps}.json"),
                orient="records",
                indent=4,
            )
            win_rate = np.mean(wins).item()
            scores = np.mean(scores).item()
            accuracy = np.mean(accuracies).item()
            response_len = np.mean(
                tree.map_structure(lambda x: len(self.tokenizer.encode(x)), responses)
            )

        dist.barrier()

        win_rate = self.strategy.broadcast(win_rate)
        scores = self.strategy.broadcast(scores)
        accuracy = self.strategy.broadcast(accuracy)
        response_len = self.strategy.broadcast(response_len)
        eval_count = self.strategy.broadcast(eval_count)

        self._post_evaluate()
        return {
            "eval/rm_win_rate": win_rate,
            "eval/score": scores,
            "eval/accuracy": accuracy,
            "eval/eval_count": eval_count,
            "eval/elapse": time.time() - st_time,
            "eval/response_tok_len": response_len,
        }

    def sync_params_to_actors(self):
        st = time.time()
        self._broadcast_to_vllm()
        self.pi_beta_version += 1
        self.pi_beta_lags_behind = False
        self.weight_sync_elapse = time.time() - st

    def _broadcast_to_vllm(self):
        dist.barrier()
        if self.args.asynchronous:
            # Pooling until generation finishes.
            while True:
                time.sleep(0.1)
                actors_busy = [actor.is_generating() for actor in self.actors]
                if not any(actors_busy):
                    break

        reset_prefix_cache_futs = []
        if self.args.enable_prefix_caching and self.strategy.is_rank_0():
            reset_prefix_cache_futs = [
                actor.futures.reset_prefix_cache() for actor in self.actors
            ]

        model = self.model.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        torch.cuda.empty_cache()
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

        if reset_prefix_cache_futs:
            _ = [fut.result() for fut in reset_prefix_cache_futs]
        torch.cuda.empty_cache()
        dist.barrier()
        logging.info(f"weights @version={self.pi_beta_version} broadcasted to actors")

    def _post_learning(self):
        torch.cuda.empty_cache()
        self.pi_beta_lags_behind = True
        if self.args.vllm_sleep:
            # Wake up vLLM after training.
            st = time.time()
            torch.cuda.synchronize()
            dist.barrier()
            if self.strategy.is_rank_0():
                futs = [actor.futures.wake_up() for actor in self.actors]
                _ = [fut.result() for fut in futs]
            torch.cuda.synchronize()
            dist.barrier()
            self.vllm_wake_up_time = time.time() - st

    def _pre_learning(self):
        if self.args.vllm_sleep:
            # Let vLLM sleep before training.
            st = time.time()
            torch.cuda.synchronize()
            dist.barrier()
            if self.strategy.is_rank_0():
                futs = [actor.futures.sleep() for actor in self.actors]
                _ = [fut.result() for fut in futs]
            torch.cuda.synchronize()
            dist.barrier()
            self.vllm_go_sleep_time = time.time() - st
