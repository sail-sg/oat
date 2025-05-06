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

# Reference to https://github.com/OpenRLHF/OpenRLHF.

import logging
import os
import random
import shutil
import time
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils import safe_get_full_grad
from peft import PeftModel, get_peft_model_state_dict
from rich.pretty import pprint
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from oat.args import OATArgs
from oat.model import LLM

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


def get_strategy(args):
    if args.rnd_seed:
        logging.info("Using randomly generated seed")
        args.seed = time.time_ns() % 2**32
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        train_batch_size_per_device=getattr(args, "train_batch_size_per_device", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return args, strategy


def get_train_ds_config(
    offload,
    adam_offload=True,
    stage=2,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=False,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "allgather_bucket_size": 1e9,
        "reduce_bucket_size": 1e9,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {
            "grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"
        },
    }


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 1000,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=[
        "bias",
        "layer_norm.weight",
        "layernorm.weight",
        "norm.weight",
        "ln_f.weight",
    ],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


class DeepspeedStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """

    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        train_batch_size_per_device=1,
        train_batch_size=1,
        zero_stage=2,
        bf16=True,
        args: OATArgs = None,
    ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.train_batch_size_per_device = train_batch_size_per_device
        self.bf16 = bf16
        self.seed = seed
        self.max_norm = max_norm
        self.adam_offload = getattr(args, "adam_offload", False)
        self.zpg = getattr(args, "zpg", 1)
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", "fp32")
        # disable_trace_cache
        self.disable_trace_cache = getattr(args, "disable_trace_cache", False)
        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
        self.set_seed(self.seed)

        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.grad_acc_step = (
            self.train_batch_size // self.train_batch_size_per_device // self.world_size
        )

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, LLM):
            model = model.model
        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def backward(
        self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs
    ) -> None:
        if isinstance(model, LLM):
            model = model.model
        model.backward(loss)

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        if isinstance(model, LLM):
            model = model.model
        model.step()

    def get_gradient_norm(
        self,
        model: nn.Module,
        **_,
    ) -> None:
        if isinstance(model, LLM):
            model = model.model
        grad_norm = torch.tensor(0.0)
        for p in model.module.parameters():
            grad = safe_get_full_grad(p)
            if grad is not None:
                grad_norm += grad.norm(2).cpu() ** 2
        grad_norm = grad_norm.sqrt()
        return grad_norm

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
    ):
        # DDP only mode, replay buffers on each rank are different.
        if sampler is None:
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
            )

        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, LLM):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                # Prepare online model
                assert (
                    len(arg) == 3
                ), f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._ds_init_train_model(*arg))
            else:
                # Prepare reference model
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(self, model, optim, scheduler):
        is_wrapped = isinstance(model, LLM)
        ds_config = self.get_ds_train_config(is_wrapped)

        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_wrapped else model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            args={"local_rank": self.args.local_rank},
            dist_init_required=True,
        )
        if is_wrapped:
            model.model = engine
        else:
            model = engine

        return model, optim, scheduler

    def get_ds_train_config(self, is_wrapped):
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=self.stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            disable_trace_cache=self.disable_trace_cache,
        )

        ds_config["train_micro_batch_size_per_gpu"] = self.train_batch_size_per_device
        train_batch_size = self.train_batch_size
        ds_config["train_batch_size"] = train_batch_size

        return ds_config

    def _ds_init_eval_model(self, model):
        is_wrapped = isinstance(model, LLM)
        ds_config = self.get_ds_eval_config(offload=self.args.ref_offload)

        engine, *_ = deepspeed.initialize(
            model=model.model if is_wrapped else model,
            args={"local_rank": self.args.local_rank},
            config=ds_config,
            dist_init_required=True,
        )
        if is_wrapped:
            model.model = engine
        else:
            model = engine
        return model

    def get_ds_eval_config(self, offload=False):
        # DS Config
        ds_config = get_eval_ds_config(
            offload=offload, stage=self.stage if self.stage == 3 else 0, bf16=self.bf16
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.train_batch_size_per_device
        ds_config["train_batch_size"] = self.train_batch_size

        return ds_config

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.grad_acc_step == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            data = param.data.to(device)
                            param_ema.data.copy_(
                                (1 - beta) * data + beta * param_ema.data
                            )
                        else:
                            # TODO: use prefiltering for efficiency
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            with deepspeed.zero.GatheredParameters(
                                params_to_fetch, enabled=len(params_to_fetch) > 0
                            ):
                                data = param.data.to(device)
                                param_ema.data.copy_(
                                    (1 - beta) * data + beta * param_ema.data
                                )

    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_model(
        self,
        model: nn.Module,
        tokenizer,
        output_dir,
        tag,
        max_num=3,
        max_mem=1000,
        **kwargs,
    ) -> None:
        if self.is_rank_0():
            save_dir = os.path.join(output_dir, tag)
            os.makedirs(save_dir, exist_ok=True)

            # max hard drive space limit
            MAX_SIZE = max_mem * 1024 * 1024 * 1024

            while True:
                # Get all subdirectory and modification time
                subdirs = [
                    (
                        os.path.join(output_dir, d),
                        os.path.getmtime(os.path.join(output_dir, d)),
                    )
                    for d in os.listdir(output_dir)
                    if os.path.isdir(os.path.join(output_dir, d))
                ]
                # Sort by modification time, oldest first
                subdirs.sort(key=lambda x: x[1])
                # Calculate the total size of all sub -directory
                total_size = 0
                for subdir, _ in subdirs:
                    for dirpath, dirnames, filenames in os.walk(subdir):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)

                # If the number of subdire directors is greater than equal to max_num or the total size is greater than max_mem, the oldest Checkpoint is deleted
                if len(subdirs) > max_num or total_size > MAX_SIZE:
                    oldest_dir, _ = subdirs[0]  # The oldest directory
                    if os.path.exists(oldest_dir):  # Ensure that the directory exists
                        shutil.rmtree(oldest_dir)  # Delete directory
                        self.print(
                            f"Deleted oldest ckpt {oldest_dir}"
                        )  # The standard print function is used here
                else:
                    break

        dist.barrier()

        # save model weights for ZeRO2/3
        model_to_save = self._unwrap_model(model)

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(
                params_to_fetch, enabled=len(params_to_fetch) > 0
            ):
                vv = v.data.cpu()
                if self.is_rank_0():
                    output_state_dict[k] = vv

        if self.is_rank_0():
            state_dict = model_to_save.state_dict()

            # copy named_buffers with `persistent=True`
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False):
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(save_dir, **kwargs)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(save_dir, "adapter_model.bin"),
                    )
            else:
                # save model
                model_to_save.save_pretrained(
                    save_dir, state_dict=output_state_dict, **kwargs
                )

            # save config
            output_config_file = os.path.join(save_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            # save tokenizer
            tokenizer.save_pretrained(save_dir)

            # for models not in AutoModel, copy python module files
            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for filename in os.listdir(train_from_model_path):
                    if filename.endswith(".py"):
                        shutil.copy(
                            os.path.join(train_from_model_path, filename),
                            os.path.join(save_dir, filename),
                        )

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(
                data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM
            )
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [
                torch.zeros_like(data).to(torch.cuda.current_device())
                for _ in range(self.world_size)
            ]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [
                torch.zeros_like(data).to(torch.cuda.current_device())
                for _ in range(self.world_size)
            ]
            dist.gather(
                data.to(torch.cuda.current_device()), ret if self.is_rank_0() else None
            )
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def broadcast(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.broadcast(v)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"
            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            dist.broadcast(data, 0)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def debug(self, *msg):
        if self.args.debug:
            print(*msg)

    def print(self, *msg):
        if self.is_rank_0():
            print("\n")
            print(*msg)
            print("\n")

    def pprint(self, *msg):
        if self.is_rank_0():
            print("\n")
            pprint(*msg)
            print("\n")

    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank()

    def save_ckpt(
        self,
        model,
        save_dir,
        tag=None,
        max_num=3,
        max_mem=1000,
        client_state={},
        save_latest=True,
    ):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        if self.is_rank_0():
            os.makedirs(save_dir, exist_ok=True)
            MAX_SIZE = max_mem * 1024**3  # Convert GB to bytes

            while True:
                subdirs = sorted(
                    [
                        (
                            os.path.join(save_dir, d),
                            os.path.getmtime(os.path.join(save_dir, d)),
                        )
                        for d in os.listdir(save_dir)
                        if os.path.isdir(os.path.join(save_dir, d))
                    ],
                    key=lambda x: x[1],
                )
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for subdir, _ in subdirs
                    for dirpath, _, filenames in os.walk(subdir)
                    for f in filenames
                )

                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir = subdirs[0][0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        dist.barrier()
        model.save_checkpoint(
            save_dir, tag=tag, client_state=client_state, save_latest=save_latest
        )

    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        load_path, states = model.load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
        )
        if load_path is None:
            raise RuntimeError(f"failed to resume from checkpoint {load_dir}")
        return load_path, states


class DummyStrategy:
    def __init__(self, args) -> None:
        self.args = args

    def print(self, *args):
        print(*args)

    def is_rank_0(self):
        return True
