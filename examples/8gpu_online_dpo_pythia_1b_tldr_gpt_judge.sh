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

python -m oat.experiment.main \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --algo DPO \
    --beta 0.1 \
    --preference-oracle gpt-4o-mini-2024-07-18 \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --input-key prompt \
    --output-key pythia-1b-reference \
    --sync-params-every 1 \
    --max-train 50000 \
    --generate-max-length 53 \
    --train-batch-size 128 \
    --rollout-batch-size 128 \
    --rollout-batch-size-per-device 32 \
    --pi-buffer-maxlen-per-device 32 \
    --train-batch-size-per-device 8 \
    --eval-steps 20 \
    --use-wb \
    --wb-run-name 1b_gpt_4o_mini_dpo_online
