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

deepspeed --module oat.experiment.run_offline \
    --gpus 8 \
    --gradient-checkpointing \
    --flash-attn \
    --rnd-seed \
    --algo DPO \
    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --apply-chat-template \
    --ref-offload \
    --zero-stage 3 \
    --beta 0.01 \
    --preference_data princeton-nlp/llama3-ultrafeedback-armorm \
    --max-train 99999 \
    --prompt_max_length 1800 \
    --generate_max_length 1000 \
    --save_steps 100 \
    --input_key prompt \
    --chosen_key chosen \
    --rejected_key rejected \
    --train_split train \
    --train_batch_size 128 \
    --train_batch_size_per_device 2 \
    --use-wb \
    --wb-run-name llama3_8b_offline_dpo \
