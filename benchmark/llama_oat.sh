# 1B: [Config 1] Collocate all three workloads.
## Actor: 8 vLLM instances each running on 1 GPU; 
## Learner: DeepSpeed zero-2 over 8 GPUs; 
## Oracle: 8 parallel RM workers each running on 1 GPU.

# 1.a) Start Mosec RM service.
python -m oat.oracles.remote.server

# 1.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash_attn \
    --rnd_seed \
    --total_gpus 8 \
    --collocate \
    --vllm_gpu_ratio 0.25 \
    --dap_algo DPO \
    --beta 0.1 \
    --zero_stage 2 \
    --reward_oracle remote \
    --remote_rm_url http://0.0.0.0:8000 \
    --remote_rm_client_workers 4 \
    --reward_oracle_batch_size 1 \
    --pretrain meta-llama/Llama-3.2-1B-Instruct \
    --prompt_data HuggingFaceH4/ultrafeedback_binarized \
    --train_split train_prefs \
    --prompt_max_length 1800 \
    --input_key prompt \
    --output_key chosen \
    --temperature 0.8 \
    --top_p 0.95 \
    --generate_max_length 1024 \
    --eval_data alpaca_eval_gpt4_baseline@tatsu-lab/alpaca_eval \
    --eval_input_key instruction \
    --eval_output_key output \
    --eval_split eval \
    --sync_params_every 1 \
    --max_train 2560 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --micro_pi_buffer_maxlen 16 \
    --micro_train_batch_size 1 \
    --eval_steps 99999 \
    --debug \
    --use_wandb True \
    --wandb_project oat-benchmark \
    --wandb_run_name 1b_llama
    
    # in the case of OOM:
    # slightly decrease --vllm_gpu_ratio, this gives space reserved for actor to learner.
    # --gradient_checkpointing
    # --ref_offload
    # --zero_stage 3
    # --adam_offload

# 2.8B: [Config 2] Collocate actors and oracle servers.
## Actor: 4 vLLM instances each running on 1 GPU (0~3); 
## Learner: DeepSpeed zero-2 over 4 GPUs (4~7); 
## Oracle: 4 parallel RM workers each running on 1 GPU (0~3).
# 2.a) Start Mosec RM service.
python -m oat.oracles.remote.server --cuda_devices 0,1,2,3

# 2.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash_attn \
    --rnd_seed \
    --total_gpus 8 \
    --vllm_gpu_ratio 0.25 \
    --dap_algo DPO \
    --beta 0.1 \
    --zero_stage 2 \
    --reward_oracle remote \
    --remote_rm_url http://0.0.0.0:8000 \
    --remote_rm_client_workers 4 \
    --reward_oracle_batch_size 1 \
    --pretrain meta-llama/Llama-3.2-1B-Instruct \
    --prompt_data HuggingFaceH4/ultrafeedback_binarized \
    --train_split train_prefs \
    --prompt_max_length 1800 \
    --input_key prompt \
    --output_key chosen \
    --temperature 0.8 \
    --top_p 0.95 \
    --generate_max_length 1024 \
    --eval_data alpaca_eval_gpt4_baseline@tatsu-lab/alpaca_eval \
    --eval_input_key instruction \
    --eval_output_key output \
    --eval_split eval \
    --sync_params_every 1 \
    --max_train 2560 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 32 \
    --micro_pi_buffer_maxlen 32 \
    --micro_train_batch_size 1 \
    --eval_steps 99999 \
    --debug \
    --use_wandb True \
    --wandb_project oat-benchmark \
    --wandb_run_name 2.8b_llama