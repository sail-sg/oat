# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 1 GPU
deepspeed --module oat.experiment.run_offline \
    --gpus 1 \
    --bf16 \
    --collocate \
    --gradient-checkpointing \
    --flash-attn \
    --algo SFT \
    --learning_rate 0.000001 \
    --pretrain Qwen/Qwen2.5-Math-1.5B \
    --zero-stage 2 \
    --chat_data "robinsmits/ChatAlpaca-20K" \
    --num_prompt_epoch 2 \
    --no-extract-content \
    --apply_chat_template \
    --max-train 9999999 \
    --max_model_len 4096 \
    --save_steps -1 \
    --train_split test \
    --train_batch_size 32 \
    --train_batch_size_per_device 1 \
    --rollout_batch_size 32 \
    --rollout_batch_size_per_device 32 \
    --pi_buffer_maxlen_per_device 32 \
    --eval_steps -1 \
    --use_fused_lm_head \
    --use_wb \
    --wb_project oat-dev \
    --wb_run_name mt-sft-debug-bs1-use_fuse
