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

# Notes -----------------------------------------------
# We assume the data is in the messages format:
# [
#   {"role": "user", "content": "user's prompt"},
#   {"role": "assistant", "content": "llm output"},
#   ...
# ]

cd /home/aiops/liuzc
source ./.zshrc
conda activate oat
cd oat

# Hyperparameter ---------------------------------------
GPUS=4
BATCH_SIZE=128
BATCH_SIZE_PER_DEVICE=4
MODEL=Qwen/Qwen2.5-Math-1.5B
DATASET=robinsmits/ChatAlpaca-20K
DATASET_KEY=messages
N_EPOCH=2

deepspeed --module oat.experiment.run_offline \
    --gpus $GPUS \
    --bf16 \
    --collocate \
    --gradient-checkpointing \
    --flash-attn \
    --algo SFT \
    --learning_rate 0.000001 \
    --pretrain $MODEL \
    --zero-stage 2 \
    --chat_data $DATASET \
    --msg_key $DATASET_KEY \
    --num_prompt_epoch $N_EPOCH \
    --max-train 9999999 \
    --max_model_len 4096 \
    --save_steps -1 \
    --train_split train \
    --train_batch_size $BATCH_SIZE \
    --train_batch_size_per_device $BATCH_SIZE_PER_DEVICE \
    --rollout_batch_size $BATCH_SIZE \
    --rollout_batch_size_per_device $(( $BATCH_SIZE / $GPUS )) \
    --pi_buffer_maxlen_per_device $(( $BATCH_SIZE / $GPUS )) \
    --eval_steps -1 \
    --no-use_fused_lm_head \
    --use_wb \
    --wb_run_name SFT-${MODEL}-${DATASET}
