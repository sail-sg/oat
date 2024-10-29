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
import argparse
import math

import torch

from oat.types import DAPAlgo


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rnd_seed", action="store_true")
    parser.add_argument("--run_name", type=str, default="default")

    parser.add_argument("--pretrain", type=str, default="google/gemma-2b")
    parser.add_argument("--ref_pretrain", type=str, default=None)

    # Resource config
    parser.add_argument(
        "--vllm_gpu_ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--total_gpus",
        type=int,
        default=8,
    )
    parser.add_argument("--collocate", action="store_true")
    parser.add_argument(
        "--shm_size_mb",
        type=int,
        default=5000,
    )

    # Prompts dataset
    parser.add_argument(
        "--prompt_data", type=str, default="lkevinzc/tldr-with-sft-reference"
    )
    parser.add_argument("--input_key", type=str, default="prompt")
    parser.add_argument("--output_key", type=str, default="output")

    parser.add_argument(
        "--eval_data", type=str, default="", help="Defaults to prompt_data if empty"
    )
    parser.add_argument(
        "--eval_input_key", type=str, default="", help="Defaults to input_key if empty"
    )
    parser.add_argument(
        "--eval_output_key",
        type=str,
        default="",
        help="Defaults to output_key if empty",
    )

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--max_train", type=int, default=50000)
    parser.add_argument("--max_queries", type=int, default=50000)
    parser.add_argument("--max_test", type=int, default=1000)

    # Offline preference dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="OpenLLMAI/preference_dataset_mixture2_and_safe_pku",
    )
    parser.add_argument(
        "--dataset_probs", type=str, default="1.0", help="sampling probs for datasets"
    )

    # Online DAP
    parser.add_argument(
        "--dap_algo",
        type=str,
        choices=["DPO", "IPO", "SLiC", "SimPO"],
        default="DPO",
        help="Direct alignment from preference method.",
    )
    parser.add_argument(
        "--sync_params_every",
        type=int,
        default=1,
        help="Set 1 for truly online; large number for offline.",
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0
    )  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument(
        "--gamma_beta_ratio", type=float, default=0.5
    )  # SimPO https://arxiv.org/pdf/2405.14734

    # Oracle
    parser.add_argument("--reward_oracle", type=str, default="pairrm")
    parser.add_argument("--reward_oracle_batch_size", type=int, default=1)
    parser.add_argument("--remote_rm_url", type=str, default="")
    parser.add_argument("--remote_rm_client_workers", type=int, default=4)
    parser.add_argument("--bt_sample", action="store_true")

    ## Epistemic reward model
    parser.add_argument("--num_ensemble", type=int, default=20)
    parser.add_argument("--enn_max_try", type=int, default=-1)
    parser.add_argument("--enn_lambda", type=float, default=0.5)
    parser.add_argument("--rm_lr", type=float, default=1e-3)
    parser.add_argument("--rm_wd", type=float, default=5e-5)
    parser.add_argument("--rm_hidden_dim", type=int, default=128)
    parser.add_argument("--rm_act_fn", type=str, default="relu")
    parser.add_argument("--rm_sgd_steps", type=int, default=5)
    parser.add_argument("--rm_fixed_reg", action="store_true")
    parser.add_argument("--rm_train_budget", type=int, default=-1)

    # Exploration
    parser.add_argument("--rm_backbone", type=str, default="llm-blender/PairRM-hf")
    parser.add_argument("--learn_rm", action="store_true")
    parser.add_argument("--learn_rm_only", action="store_true")

    # Model-based
    parser.add_argument("--model_rollout", action="store_true")
    parser.add_argument("--max_model_data_ratio", type=float, default=0.3)
    parser.add_argument(
        "--model_data_strategy",
        type=str,
        choices=["random"],
        default="random",
        help="Dyna search control.",
    )
    parser.add_argument("--burn_in_period", type=int, default=5)
    parser.add_argument("--pure_model_based", action="store_true")

    parser.add_argument(
        "--exp_method",
        type=str,
        choices=[
            "no",
            "EnnBAITS",
            "EnnEETS",
            "EnnUncertainty",
            "EnnPassive",
        ],
        default="no",
        help="Types of exploration.",
    )
    parser.add_argument("--exp_pretrain", type=str, default="")
    parser.add_argument("--exp_rnd_sample", action="store_true")
    parser.add_argument("--exp_allow_second_best", action="store_true")

    # Evaluation params
    parser.add_argument("--online_evaluation", action="store_true")
    parser.add_argument("--best_of_n_eval", action="store_true")
    parser.add_argument("--num_bon", type=int, default=1)
    parser.add_argument("--bon_temperature", type=float, default=0.7)
    parser.add_argument("--eval_batch_size", type=int, default=-1)
    parser.add_argument("--eval_generate_max_length", type=int, default=200)
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--eval_top_p", type=float, default=0.95)
    parser.add_argument("--eval_top_k", type=float, default=-1)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--eval_query_interval", type=int, default=-1)

    # Generation params
    parser.add_argument("--generate_max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=float, default=-1)
    parser.add_argument("--num_samples", type=int, default=2)

    # Training specs
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--num_prompt_epoch", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--rollout_batch_size", type=int, default=32)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_sgd_steps", type=int, default=999999999)
    parser.add_argument("--micro_pi_buffer_maxlen", type=int, default=8)
    parser.add_argument("--r_buffer_maxlen", type=int, default=50000)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--max_step_adjustment", type=float, default=1)
    parser.add_argument("--buffer_clear_every", type=int, default=999999999)
    parser.add_argument("--dump_all_buffer", action="store_true")

    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for deepspeed"
    )
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")

    parser.add_argument("--input_template", type=str, default="")
    parser.add_argument("--apply_chat_template", action="store_true", default=False)

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="oat-llm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="debug",
    )
    return parser


def default_args_validation(args: argparse.Namespace):
    # Validation.
    args.dap_algo = getattr(DAPAlgo, args.dap_algo)
    if args.dap_algo != DAPAlgo.SimPO and (
        args.ref_pretrain is None or args.ref_pretrain == ""
    ):
        args.ref_pretrain = args.pretrain
    if args.learn_rm:
        assert args.exp_method != "no" and args.exp_pretrain == ""
    if args.learn_rm_only:
        assert args.best_of_n_eval
    if args.enn_max_try == -1:
        args.enn_max_try = args.num_ensemble
    if args.eval_batch_size == -1:
        args.eval_batch_size = args.micro_rollout_batch_size
    if args.rm_train_budget == -1:
        args.rm_train_budget = math.inf
    args.max_queries = max(args.max_queries, args.max_train)
    args.max_model_len = (
        args.prompt_max_length
        + max(args.generate_max_length, args.eval_generate_max_length)
        + 128
    )
    gpu_available = torch.cuda.device_count()
    assert (
        gpu_available >= args.total_gpus
    ), f"{gpu_available} GPUs available, but {args.total_gpus} required"
    return args
