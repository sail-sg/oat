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

# Dependencies for benchmarking to use *nccl*:
# vllm==0.4.2, transformers==4.43.3, flash-attn==2.5.8, deepspeed==0.14.4

# 1B: [Config 1] Collocate all three workloads.
## Actor: 8 vLLM instances each running on 1 GPU; 
## Learner: DeepSpeed zero-2 over 8 GPUs; 
## Oracle: 8 parallel RM workers each running on 1 GPU.
# 1.a) Start Mosec RM service.
python benchmark/pythia_custom_remote_rm.py --remote_rm_model trl-lib/pythia-1b-deduped-tldr-rm
# 1.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash_attn \
    --rnd_seed \
    --total_gpus 8 \
    --collocate \
    --dap_algo DPO \
    --beta 0.1 \
    --reward_oracle remote \
    --remote_rm_url http://0.0.0.0:8000 \
    --remote_rm_client_workers 8 \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt_data lkevinzc/tldr-with-sft-reference \
    --input_key prompt \
    --output_key pythia-1b-reference \
    --sync_params_every 1 \
    --max_train 2560 \
    --generate_max_length 53 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --micro_pi_buffer_maxlen 16 \
    --micro_train_batch_size 8 \
    --eval_steps 99999 \
    --debug \
    --use_wandb True \
    --wandb_project oat-benchmark \
    --wandb_run_name 1b_pythia


# 1B: [Config 2] Collocate actors and oracle servers.
## Actor: 4 vLLM instances each running on 1 GPU (0~3); 
## Learner: DeepSpeed zero-2 over 4 GPUs (4~7); 
## Oracle: 4 parallel RM workers each running on 1 GPU (0~3).
# 1.2.a) Start Mosec RM service.
python benchmark/pythia_custom_remote_rm.py --remote_rm_model trl-lib/pythia-1b-deduped-tldr-rm --cuda_devices 0,1,2,3
# 1.2.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash_attn \
    --rnd_seed \
    --total_gpus 8 \
    --dap_algo DPO \
    --beta 0.1 \
    --reward_oracle remote \
    --remote_rm_url http://0.0.0.0:8000 \
    --remote_rm_client_workers 8 \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt_data lkevinzc/tldr-with-sft-reference \
    --input_key prompt \
    --output_key pythia-1b-reference \
    --sync_params_every 1 \
    --max_train 2560 \
    --generate_max_length 53 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 32 \
    --micro_pi_buffer_maxlen 32 \
    --micro_train_batch_size 8 \
    --eval_steps 99999 \
    --debug \
    --use_wandb True \
    --wandb_project oat-benchmark \
    --wandb_run_name 1b_pythia


# 2.8B: [Config 2] Collocate actors and oracle servers.
## Actor: 4 vLLM instances each running on 1 GPU (0~3); 
## Learner: DeepSpeed zero-2 over 4 GPUs (4~7); 
## Oracle: 4 parallel RM workers each running on 1 GPU (0~3).
# 2.a) Start Mosec RM service.
python benchmark/pythia_custom_remote_rm.py --remote_rm_model trl-lib/pythia-2.8b-deduped-tldr-rm --cuda_devices 0,1,2,3
# 2.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash_attn \
    --rnd_seed \
    --total_gpus 8 \
    --vllm_gpu_ratio 0.35 \
    --zero_stage 2 \
    --dap_algo DPO \
    --beta 0.1 \
    --reward_oracle remote \
    --remote_rm_url http://0.0.0.0:8000 \
    --remote_rm_client_workers 8 \
    --pretrain trl-lib/pythia-2.8b-deduped-tldr-sft \
    --prompt_data lkevinzc/tldr-with-sft-reference \
    --input_key prompt \
    --output_key pythia-2.8b-reference \
    --sync_params_every 1 \
    --max_train 2560 \
    --generate_max_length 53 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 32 \
    --micro_pi_buffer_maxlen 32 \
    --micro_train_batch_size 2 \
    --eval_steps 99999 \
    --debug \
    --use_wandb True \
    --wandb_project oat-benchmark \
    --wandb_run_name 2.8b_pythia


# 6.9B: [Config 2] Collocate actors and oracle servers.
## Actor: 4 vLLM instances each running on 1 GPU (0~3); 
## Learner: DeepSpeed zero-2 over 4 GPUs (4~7); 
## Oracle: 4 parallel RM workers each running on 1 GPU (0~3).
# 3.a) Start Mosec RM service.
python benchmark/pythia_custom_remote_rm.py --remote_rm_model trl-lib/pythia-6.9b-deduped-tldr-rm --tokenizer trl-lib/pythia-6.9b-deduped-tldr-sft --cuda_devices 0,1,2,3
# 3.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash_attn \
    --ref_offload \
    --gradient_checkpointing \
    --rnd_seed \
    --total_gpus 8 \
    --vllm_gpu_ratio 0.55 \
    --zero_stage 2 \
    --adam_offload \
    --dap_algo DPO \
    --beta 0.1 \
    --reward_oracle remote \
    --remote_rm_url http://0.0.0.0:8000 \
    --remote_rm_client_workers 8 \
    --pretrain trl-lib/pythia-6.9b-deduped-tldr-sft \
    --prompt_data lkevinzc/tldr-with-sft-reference \
    --input_key prompt \
    --output_key pythia-6.9b-reference \
    --sync_params_every 1 \
    --max_train 2560 \
    --generate_max_length 53 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 32 \
    --micro_pi_buffer_maxlen 32 \
    --micro_train_batch_size 4 \
    --eval_steps 99999 \
    --debug \
    --use_wandb True \
    --wandb_project oat-benchmark \
    --wandb_run_name 6.9b_pythia
