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

# 1B
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --output_dir pythia-1b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_steps 20 \
    --max_new_tokens 53 \
    --temperature 0.7 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --logging_steps 1 \
    --save_steps 9999 \
    --eval_steps 9999

# 2B
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-2.8b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-2.8b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --output_dir pythia-2.8b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_steps 20 \
    --max_new_tokens 53 \
    --temperature 0.7 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --logging_steps 1 \
    --save_steps 9999 \
    --eval_steps 9999

# 7B
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-6.9b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-6.9b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --output_dir pythia-6.9b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 20 \
    --max_new_tokens 53 \
    --temperature 0.7 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --logging_steps 1 \
    --save_steps 9999 \
    --eval_steps 9999
