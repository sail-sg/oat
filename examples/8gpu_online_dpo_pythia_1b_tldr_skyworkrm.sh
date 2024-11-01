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
python -m oat.oracles.remote.server --cuda_devices 0,1,2,3

# 2) Start the actor and learner. 
# Change `--sync_params_every` to a large number (e.g., 999999) for Offline.
python -m oat.experiment.main \
    --flash_attn \
    --gradient_checkpointing \
    --rnd_seed \
    --total_gpus 8 \
    --dap_algo DPO \
    --reward_oracle remote \
    --remote_rm_url http://0.0.0.0:8000 \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --beta 0.1 --prompt_data lkevinzc/tldr-with-sft-reference \
    --input_key prompt \
    --output_key pythia-1b-reference \
    --sync_params_every 999999 \
    --max_train 50000 \
    --generate_max_length 53 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 32 \
    --micro_pi_buffer_maxlen 32 \
    --micro_train_batch_size 8 \
    --eval_steps 20 \
    --use_wandb True \
    --wandb_run_name 1b_skywork_dpo_offline
