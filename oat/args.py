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
from typing import Literal, Optional

import torch
import tyro

from oat.types import DAPAlgo, RLAlgo, SFTAlgo


@dataclass
class OATArgs:
    """Experiment arguments."""

    """Resources."""
    # Launchpad launch type
    launch_type: str = "local_mp"
    # Number of GPUs to run the experiment.
    gpus: int = 8
    # Ratio of pre-allocated GPU memory for vLLM.
    vllm_gpu_ratio: float = 0.25
    # Max model length.
    max_model_len: Optional[int] = None
    # Enable vLLM prefix caching.
    enable_prefix_caching: bool = False
    # Actor-learner collocation.
    collocate: bool = False
    # Offload vLLM weights & discard KV cache to collocate larger models.
    vllm_sleep: bool = False
    # Size of Plasma shared memory.
    shm_size_mb: int = 5000
    # Asynchronous training.
    asynchronous: bool = False

    """Training configurations."""
    # Model name.
    pretrain: str = "trl-lib/pythia-1b-deduped-tldr-sft"
    # Reference model name, defaults to pretrain if None.
    ref_pretrain: str = None
    # Critic initial model.
    critic_pretrain: str = None
    # Tokenizer name.
    tokenizer: str = ""

    # LLM alignment algorithms.
    algo: Literal[
        "DPO",
        "IPO",
        "LR_DPO",
        "SLiC",
        "SimPO",
        "BNF",
        "SFT",
        "PPO",
    ] = "DPO"
    # Set 1 for truly online algorithms; large number for offline; intermediate value for iterative.
    sync_params_every: int = 1
    # Used in KL-regularized losses.
    beta: float = 0.1
    # cDPO https://arxiv.org/pdf/2305.18290.
    label_smoothing: float = 0
    # SimPO https://arxiv.org/pdf/2405.14734.
    gamma_beta_ratio: float = 0.5
    # DPO-Positive https://arxiv.org/pdf/2402.13228.
    dpo_positive_lambda: float = 0
    # DPO + SFT loss coefficient.
    sft_weight: float = 0

    # Oracle.
    oracle: str = "pairrm"
    oracle_type: Literal["preference", "reward"] = "preference"
    oracle_batch_size: int = 1
    remote_rm_url: str = ""
    remote_rm_client_workers: int = 4
    # Sampling a Bernoulli to get the binary feedback instead of thresholding.
    bt_sample: bool = False

    # Critic.
    critic_type: Literal["ppo", "grpo", "drgrpo"] = "drgrpo"

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
    generate_max_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    num_samples: int = 2

    """Evaluation configurations."""
    online_evaluation: bool = False
    best_of_n_eval: bool = False
    num_bon: int = 1
    bon_temperature: float = 0.7
    max_eval: int = 1000
    eval_split: str = "test"
    eval_batch_size: int = -1
    eval_generate_max_length: int = 512
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_top_k: int = -1
    eval_n: int = 1
    eval_steps: int = 20
    eval_query_interval: int = -1
    # Defaults to prompt_data if empty.
    eval_data: str = ""
    # Defaults to input_key if empty.
    eval_input_key: str = ""
    # Defaults to output_key if empty.
    eval_output_key: str = ""

    """Training specs."""
    save_path: str = "./oat-output"
    save_steps: int = -1
    save_from: int = 0
    save_ckpt: bool = False
    resume_dir: str = ""
    resume_tag: str = None
    max_save_num: int = 5
    max_save_mem: int = 1000
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
    critic_max_step_adjustment: float = 1
    buffer_clear_every: float = math.inf
    dump_all_buffer: bool = False
    dump_replay_every: int = -1

    max_norm: float = 1.0
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.95
    l2: float = 0.0
    gradient_checkpointing: bool = False
    seed: int = 42
    disable_fast_tokenizer: bool = False
    local_rank: int = -1

    zero_stage: int = 2
    bf16: bool = True
    ref_offload: bool = False
    learning_rate: float = 5e-7
    critic_learning_rate: float = 9e-6
    lr_scheduler: Literal["cosine_with_min_lr", "polynomial", "constant"] = (
        "cosine_with_min_lr"
    )
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


def get_default_args(args_cls=OATArgs):
    return tyro.cli(args_cls)


def default_args_validation(args: OATArgs):
    # Validation.
    for algo_pool in [DAPAlgo, RLAlgo, SFTAlgo]:
        try:
            args.algo = getattr(algo_pool, args.algo)
            break
        except AttributeError:
            continue
    else:
        raise ValueError(f"Invalid algorithm name {args.algo}")

    if args.algo != DAPAlgo.SimPO and (
        args.ref_pretrain is None or args.ref_pretrain == ""
    ):
        args.ref_pretrain = args.pretrain
    if args.critic_pretrain is None:
        args.critic_pretrain = args.pretrain
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
    if args.asynchronous:
        assert not args.collocate, "async training needs to disable collocation"
    gpu_available = torch.cuda.device_count()
    assert (
        gpu_available >= args.gpus
    ), f"{gpu_available} GPUs available, but {args.gpus} required"
    return args
