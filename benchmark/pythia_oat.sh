

# 1B: Fully parallel.
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
    --reward_oracle_batch_size 1 \
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

# 2.8B
# 2.a) Start Mosec RM service.
python benchmark/pythia_custom_remote_rm.py --remote_rm_model trl-lib/pythia-2.8-deduped-tldr-rm

# 2.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash_attn \
    --rnd_seed \
    --total_gpus 8 \
    --collocate \
    --dap_algo DPO \
    --beta 0.1 \
    --reward_oracle remote \
    --remote_rm_url http://0.0.0.0:8000 \
    --pretrain trl-lib/pythia-2.8b-deduped-tldr-sft \
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
    --wandb_run_name 2.8b_pythia

