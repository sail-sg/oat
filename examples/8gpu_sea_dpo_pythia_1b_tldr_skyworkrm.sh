# NOTE: Please run step 1 first then step 2 (in two bash sessions).
# 1) Start the RM service locally.
python -m oat.oracles.remote.server --cuda_devices 0,1,2,3

# 2) Start the actor and learner. 
python -m oat.experiment.main \
    --flash_attn \
    --gradient_checkpointing \
    --rnd_seed \
    --total_gpus 8 \
    --dap_algo DPO \
    --reward_oracle remote \
    --remote_rm_url http://remote-rm \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --beta 0.1 --prompt_data lkevinzc/tldr-with-sft-reference \
    --input_key prompt \
    --output_key pythia-1b-reference \
    --sync_params_every 1 \
    --num_prompt_epoch 2 \
    --max_train 50000 \
    --max_step_adjustment 0.75 \
    --lr_warmup_ratio 0.02 \
    --generate_max_length 53 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 32 \
    --micro_pi_buffer_maxlen 32 \
    --micro_train_batch_size 8 \
    --eval_steps 20 \
    --eval_query_interval 2560 \
    --num_samples 20 \
    --learn_rm \
    --exp_method EnnBAITS \
    --exp_rnd_sample \
    --model_rollout \
    --max_model_data_ratio 0.3 \
    --use_wandb True \
    --wandb_run_name 1b_skywork_dpo_sea