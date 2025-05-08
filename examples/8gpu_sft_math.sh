# Copyright 2025 Garena Online Private Limited
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

deepspeed --module oat.experiment.run_offline \
    --gpus 8 \
    --collocate \
    --gradient-checkpointing \
    --flash-attn \
    --rnd-seed \
    --algo SFT \
    --learning_rate 0.000001 \
    --pretrain Qwen/Qwen2.5-Math-1.5B \
    --zero-stage 2 \
    --preference_data lkevinzc/math-collection \
    --num_prompt_epoch 2 \
    --no-extract-content \
    --apply_chat_template \
    --max-train 9999999 \
    --prompt_max_length 2048 \
    --generate_max_length 4096 \
    --save_steps 500 \
    --prompt_key problem \
    --chosen_key solution \
    --rejected_key solution \
    --train_split train \
    --train_batch_size 128 \
    --train_batch_size_per_device 4 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --eval_steps -1 \
    --use-wb \
    --wb_project oat-math \
    --wb-run-name Qwen2.5-Math-1.5B
