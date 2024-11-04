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
"""Argument parsing."""
import math
from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from oat.types import DAPAlgo


@dataclass
class OATArgs:
    """Experiment arguments."""

    """Resources."""
    # Number of GPUs to run the experiment.
    gpus: int = 8
    # Ratio of pre-allocated GPU memory for vLLM.
    vllm_gpu_ratio: float = 0.25
    # Actor-learner collocation.
    collocate: bool = False
    # Size of Plasma shared memory.
    shm_size_mb: int = 5000

    """Training configurations."""
    # Model name.
    pretrain: str = "trl-lib/pythia-1b-deduped-tldr-sft"
    # Reference model name, defaults to pretrain if None.
    ref_pretrain: str = None

    # Direct alignment from preference methods.
    dap_algo: Literal["DPO", "IPO", "SLiC", "SimPO"] = "DPO"
    # Set 1 for truly online DAP; large number for offline.
    sync_params_every: int = 1
    # Used in DAP losses.
    beta: float = 0.1
    # cDPO https://arxiv.org/pdf/2305.18290.pdf.
    label_smoothing: float = 0
    # SimPO https://arxiv.org/pdf/2405.14734.
    gamma_beta_ratio: float = 0.5

    # Oracle.
    reward_oracle: str = "pairrm"
    reward_oracle_batch_size: int = 1
    remote_rm_url: str = ""
    remote_rm_client_workers: int = 4
    # Sampling a Bernoulli to get the binary feedback instead of thresholding.
    bt_sample: bool = False

    # Epistemic reward model (for exploration).
    num_ensemble: int = 20
    enn_max_try: int = -1
    enn_lambda: float = 0.5
    learn_rm: bool = False
    rm_lr: float = 1e-3
    rm_wd: float = 5e-5
    rm_hidden_dim: int = 128
    rm_act_fn: str = "relu"
    rm_sgd_steps: int = 5
    rm_fixed_reg: bool = False
    rm_train_budget: int = -1
    rm_backbone: str = "llm-blender/PairRM-hf"
    # Learn the ERM only without updating the LLM.
    learn_rm_only: bool = False
    # Load a pre-trained RM.
    rm_pretrain: str = ""
    # Exploration strategies.
    exp_method: Literal[
        "no",
        "EnnBAITS",
        "EnnEETS",
        "EnnUncertainty",
        "EnnPassive",
    ] = "no"
    # Random sampling if the dueling responses coincide.
    exp_rnd_sample: bool = False
    # Take the top 2 best actions.
    exp_allow_second_best: bool = False
    # Enable SEA's Mixed Preference Learning (Dyna)
    model_rollout: bool = False
    max_model_data_ratio: float = 0.3
    burn_in_period: int = 5
    pure_model_based: bool = False
    # Dyna search control.
    model_data_strategy: Literal["random"] = "random"

    # Prompt dataset.
    prompt_data: str = "lkevinzc/tldr-with-sft-reference"
    input_key: str = "prompt"
    output_key: str = "output"
    train_split: str = "train"
    max_train: int = 50000
    # Maximum number of oracle queries, defaults to max_train.
    max_queries: int = -1

    # On-policy generation params.
    generate_max_length: int = 53
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: float = -1
    num_samples: int = 2

    """Evaluation configurations."""
    online_evaluation: bool = False
    best_of_n_eval: bool = False
    num_bon: int = 1
    bon_temperature: float = 0.7
    max_eval: int = 1000
    eval_split: str = "test"
    eval_batch_size: int = -1
    eval_generate_max_length: int = 200
    eval_temperature: float = 0.0
    eval_top_p: float = 0.95
    eval_top_k: float = -1
    eval_steps: int = 20
    eval_query_interval: int = -1
    # Defaults to prompt_data if empty.
    eval_data: str = ""
    # Defaults to input_key if empty.
    eval_input_key: str = ""
    # Defaults to output_key if empty.
    eval_output_key: str = ""

    """Training specs."""
    save_path: str = "./output"
    save_steps: int = -1
    logging_steps: int = 1
    num_prompt_epoch: int = 1
    train_batch_size: int = 128
    train_batch_size_per_device: int = 1
    rollout_batch_size: int = 128
    rollout_batch_size_per_device: int = 16
    pi_buffer_maxlen_per_device: int = 16
    max_epochs: int = 1
    max_sgd_steps: float = math.inf
    r_buffer_maxlen: int = 50000
    prompt_max_length: int = 1024
    max_step_adjustment: float = 1
    buffer_clear_every: float = math.inf
    dump_all_buffer: bool = False

    max_norm: float = 1.0
    l2: float = 0.0
    gradient_checkpointing: bool = False
    seed: int = 42
    disable_fast_tokenizer: bool = False
    local_rank: int = -1

    zero_stage: int = 2
    bf16: bool = True
    ref_offload: bool = False
    learning_rate: float = 5e-7
    lr_warmup_ratio: float = 0.03
    zpg: int = 1
    adam_offload: bool = False
    flash_attn: bool = True
    grad_accum_dtype: str = None
    disable_trace_cache: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str = "all-linear"
    lora_dropout: float = 0
    gradient_checkpointing_use_reentrant: bool = False

    apply_chat_template: bool = False

    """Misc."""
    # Skip the first evaluation.
    debug: bool = False
    # Random seed conditioned on time.
    rnd_seed: bool = True

    # Weights and biases logging.
    use_wb: bool = False
    wb_org: str = None
    wb_group: str = None
    wb_project: str = "oat-llm"
    wb_run_name: str = "debug"


# def get_default_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--debug", action="store_true")
#     parser.add_argument("--rnd_seed", action="store_true")
#     parser.add_argument("--run_name", type=str, default="default")

#     parser.add_argument("--pretrain", type=str, default="google/gemma-2b")
#     parser.add_argument("--ref_pretrain", type=str, default=None)

#     # Resource config
#     parser.add_argument(
#         "--vllm_gpu_ratio",
#         type=float,
#         default=0.25,
#     )
#     parser.add_argument(
#         "--gpus",
#         type=int,
#         default=8,
#     )
#     parser.add_argument("--collocate", action="store_true")
#     parser.add_argument(
#         "--shm_size_mb",
#         type=int,
#         default=5000,
#     )

#     # Prompts dataset
#     parser.add_argument(
#         "--prompt_data", type=str, default="lkevinzc/tldr-with-sft-reference"
#     )
#     parser.add_argument("--input_key", type=str, default="prompt")
#     parser.add_argument("--output_key", type=str, default="output")

#     parser.add_argument(
#         "--eval_data", type=str, default="", help="Defaults to prompt_data if empty"
#     )
#     parser.add_argument(
#         "--eval_input_key", type=str, default="", help="Defaults to input_key if empty"
#     )
#     parser.add_argument(
#         "--eval_output_key",
#         type=str,
#         default="",
#         help="Defaults to output_key if empty",
#     )

#     parser.add_argument("--train_split", type=str, default="train")
#     parser.add_argument("--eval_split", type=str, default="test")
#     parser.add_argument("--max_train", type=int, default=50000)
#     parser.add_argument("--max_queries", type=int, default=50000)
#     parser.add_argument("--max_test", type=int, default=1000)

#     # Online DAP
#     parser.add_argument(
#         "--dap_algo",
#         type=str,
#         choices=["DPO", "IPO", "SLiC", "SimPO"],
#         default="DPO",
#         help="Direct alignment from preference method.",
#     )
#     parser.add_argument(
#         "--sync_params_every",
#         type=int,
#         default=1,
#         help="Set 1 for truly online; large number for offline.",
#     )
#     parser.add_argument("--beta", type=float, default=2.0)
#     parser.add_argument(
#         "--label_smoothing", type=float, default=0.0
#     )  # cDPO https://arxiv.org/pdf/2305.18290.pdf
#     parser.add_argument(
#         "--gamma_beta_ratio", type=float, default=0.5
#     )  # SimPO https://arxiv.org/pdf/2405.14734

#     # Oracle
#     parser.add_argument("--reward_oracle", type=str, default="pairrm")
#     parser.add_argument("--reward_oracle_batch_size", type=int, default=1)
#     parser.add_argument("--remote_rm_url", type=str, default="")
#     parser.add_argument("--remote_rm_client_workers", type=int, default=4)
#     parser.add_argument("--bt_sample", action="store_true")

#     ## Epistemic reward model
#     parser.add_argument("--num_ensemble", type=int, default=20)
#     parser.add_argument("--enn_max_try", type=int, default=-1)
#     parser.add_argument("--enn_lambda", type=float, default=0.5)
#     parser.add_argument("--rm_lr", type=float, default=1e-3)
#     parser.add_argument("--rm_wd", type=float, default=5e-5)
#     parser.add_argument("--rm_hidden_dim", type=int, default=128)
#     parser.add_argument("--rm_act_fn", type=str, default="relu")
#     parser.add_argument("--rm_sgd_steps", type=int, default=5)
#     parser.add_argument("--rm_fixed_reg", action="store_true")
#     parser.add_argument("--rm_train_budget", type=int, default=-1)

#     # Exploration
#     parser.add_argument("--rm_backbone", type=str, default="llm-blender/PairRM-hf")
#     parser.add_argument("--learn_rm", action="store_true")
#     parser.add_argument("--learn_rm_only", action="store_true")

#     # Model-based
#     parser.add_argument("--model_rollout", action="store_true")
#     parser.add_argument("--max_model_data_ratio", type=float, default=0.3)
#     parser.add_argument(
#         "--model_data_strategy",
#         type=str,
#         choices=["random"],
#         default="random",
#         help="Dyna search control.",
#     )
#     parser.add_argument("--burn_in_period", type=int, default=5)
#     parser.add_argument("--pure_model_based", action="store_true")

#     parser.add_argument(
#         "--exp_method",
#         type=str,
#         choices=[
#             "no",
#             "EnnBAITS",
#             "EnnEETS",
#             "EnnUncertainty",
#             "EnnPassive",
#         ],
#         default="no",
#         help="Types of exploration.",
#     )
#     parser.add_argument("--rm_pretrain", type=str, default="")
#     parser.add_argument("--exp_rnd_sample", action="store_true")
#     parser.add_argument("--exp_allow_second_best", action="store_true")

#     # Evaluation params
#     parser.add_argument("--online_evaluation", action="store_true")
#     parser.add_argument("--best_of_n_eval", action="store_true")
#     parser.add_argument("--num_bon", type=int, default=1)
#     parser.add_argument("--bon_temperature", type=float, default=0.7)
#     parser.add_argument("--eval_batch_size", type=int, default=-1)
#     parser.add_argument("--eval_generate_max_length", type=int, default=200)
#     parser.add_argument("--eval_temperature", type=float, default=0.0)
#     parser.add_argument("--eval_top_p", type=float, default=0.95)
#     parser.add_argument("--eval_top_k", type=float, default=-1)
#     parser.add_argument("--eval_steps", type=int, default=20)
#     parser.add_argument("--eval_query_interval", type=int, default=-1)

#     # Generation params
#     parser.add_argument("--generate_max_length", type=int, default=1024)
#     parser.add_argument("--temperature", type=float, default=0.7)
#     parser.add_argument("--top_p", type=float, default=1.0)
#     parser.add_argument("--top_k", type=float, default=-1)
#     parser.add_argument("--num_samples", type=int, default=2)

#     # Training specs
#     parser.add_argument("--save_path", type=str, default="./output")
#     parser.add_argument("--save_steps", type=int, default=-1)
#     parser.add_argument("--logging_steps", type=int, default=1)
#     parser.add_argument("--num_prompt_epoch", type=int, default=1)
#     parser.add_argument("--train_batch_size_per_device", type=int, default=1)
#     parser.add_argument("--train_batch_size", type=int, default=128)
#     parser.add_argument("--rollout_batch_size", type=int, default=128)
#     parser.add_argument("--rollout_batch_size_per_device", type=int, default=16)
#     parser.add_argument("--pi_buffer_maxlen_per_device", type=int, default=16)
#     parser.add_argument("--max_epochs", type=int, default=1)
#     parser.add_argument("--max_sgd_steps", type=int, default=999999999)
#     parser.add_argument("--r_buffer_maxlen", type=int, default=50000)
#     parser.add_argument("--prompt_max_length", type=int, default=1024)
#     parser.add_argument("--max_step_adjustment", type=float, default=1)
#     parser.add_argument("--buffer_clear_every", type=int, default=999999999)
#     parser.add_argument("--dump_all_buffer", action="store_true")

#     parser.add_argument("--load_checkpoint", action="store_true", default=False)
#     parser.add_argument("--max_norm", type=float, default=1.0)
#     parser.add_argument("--l2", type=float, default=0.0)
#     parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

#     parser.add_argument(
#         "--local_rank", type=int, default=-1, help="local_rank for deepspeed"
#     )
#     parser.add_argument("--zero_stage", type=int, default=2)
#     parser.add_argument("--bf16", action="store_true", default=True)
#     parser.add_argument("--ref_offload", action="store_true", default=False)
#     parser.add_argument("--learning_rate", type=float, default=5e-7)
#     parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
#     parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
#     parser.add_argument("--adam_offload", action="store_true", default=False)
#     parser.add_argument("--flash_attn", action="store_true", default=False)
#     parser.add_argument("--aux_loss_coef", type=float, default=0)
#     parser.add_argument("--grad_accum_dtype", type=str, default=None)
#     parser.add_argument("--disable_trace_cache", action="store_true", default=False)
#     parser.add_argument("--load_in_4bit", action="store_true", default=False)
#     parser.add_argument("--lora_rank", type=int, default=0)
#     parser.add_argument("--lora_alpha", type=int, default=16)
#     parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
#     parser.add_argument("--lora_dropout", type=float, default=0)
#     parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")

#     parser.add_argument("--input_template", type=str, default="")
#     parser.add_argument("--apply_chat_template", action="store_true", default=False)

#     # wandb parameters
#     parser.add_argument("--use_wandb", type=str, default=None)
#     parser.add_argument("--wandb_org", type=str, default=None)
#     parser.add_argument("--wandb_group", type=str, default=None)
#     parser.add_argument("--wandb_project", type=str, default="oat-llm")
#     parser.add_argument(
#         "--wandb_run_name",
#         type=str,
#         default="debug",
#     )
#     return parser


def get_default_args(args_cls=OATArgs):
    return tyro.cli(args_cls)


def default_args_validation(args: OATArgs):
    # Validation.
    args.dap_algo = getattr(DAPAlgo, args.dap_algo)
    if args.dap_algo != DAPAlgo.SimPO and (
        args.ref_pretrain is None or args.ref_pretrain == ""
    ):
        args.ref_pretrain = args.pretrain
    if args.learn_rm:
        assert args.exp_method != "no" and args.rm_pretrain == ""
    if args.learn_rm_only:
        assert args.best_of_n_eval
    if args.enn_max_try == -1:
        args.enn_max_try = args.num_ensemble
    if args.eval_batch_size == -1:
        args.eval_batch_size = args.rollout_batch_size_per_device
    if args.rm_train_budget == -1:
        args.rm_train_budget = math.inf
    if args.max_queries > 0:
        args.max_queries = min(args.max_queries, args.max_train)
    else:
        args.max_queries = args.max_train
    args.max_model_len = (
        args.prompt_max_length
        + max(args.generate_max_length, args.eval_generate_max_length)
        + 128
    )
    gpu_available = torch.cuda.device_count()
    assert (
        gpu_available >= args.gpus
    ), f"{gpu_available} GPUs available, but {args.gpus} required"
    return args
