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

# NOTE: Please run step 1 first then step 2 (in two bash sessions).
# 1) Start the RM service locally.
MOSEC_LOG_LEVEL=debug python -m oat.oracles.remote.server --cuda-devices 0,1,2,3

# 2) Start the actor and learner. 
python -m oat.experiment.main \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --input-key prompt \
    --output-key pythia-1b-reference \
    --sync-params-every 1 \
    --num-prompt-epoch 2 \
    --max-train 50000 \
    --max-step-adjustment 0.75 \
    --lr-warmup-ratio 0.02 \
    --generate-max-length 53 \
    --train-batch-size 128 \
    --rollout-batch-size 128 \
    --rollout-batch-size-per-device 32 \
    --pi-buffer-maxlen-per-device 32 \
    --train-batch-size-per-device 8 \
    --eval-steps 20 \
    --eval-query-interval 2560 \
    --num-samples 20 \
    --learn-rm \
    --exp-method EnnBAITS \
    --exp-rnd-sample \
    --model-rollout \
    --max-model-data-ratio 0.3 \
    --use-wb \
    --wb-run-name 1b_skywork_dpo_sea
