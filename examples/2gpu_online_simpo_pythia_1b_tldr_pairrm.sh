cd /home/aiops/liuzc
source ./.zshrc
conda activate ellm
python -m oat.experiment.main --flash_attn --rnd_seed --total_gpus 2 --collocate --dap_algo SimPO --reward_oracle pairrm --pretrain trl-lib/pythia-1b-deduped-tldr-sft --prompt_data lkevinzc/tldr-with-sft-reference --input_key prompt --output_key pythia-1b-reference --sync_params_every 1 --max_train 50000 --generate_max_length 53 --train_batch_size 128 --rollout_batch_size 128 --micro_rollout_batch_size 64 --micro_pi_buffer_maxlen 64 --micro_train_batch_size 8 --eval_steps 20 --use_wandb True --wandb_run_name 1b_pairrm_simpo_online
