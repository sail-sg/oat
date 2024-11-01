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

# `--collocate` means the actor and learner both use 2 GPUs.
# Change `--sync_params_every` to a large number (e.g., 999999) for Offline.
python -m oat.experiment.main \
    --rnd_seed \
    --total_gpus 2 \
    --collocate \
    --dap_algo SimPO \
    --reward_oracle pairrm \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt_data lkevinzc/tldr-with-sft-reference \
    --input_key prompt \
    --output_key pythia-1b-reference \
    --sync_params_every 1 \
    --max_train 50000 \
    --generate_max_length 53 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 64 \
    --micro_pi_buffer_maxlen 64 \
    --micro_train_batch_size 8 \
    --eval_steps 20 \
    --use_wandb True \
    --wandb_run_name 1b_pairrm_simpo_online
