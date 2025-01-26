We can easily extend oat 🌾 by running RL with rule-based rewards (results verification) to improve language model's reasoning capability. Below we show an example to run PPO on GSM8K, which improves the test score significantly **from 40.6% to 55.7% with pure RL**!

```
python -m oat.experiment.run_ppo \
    --gpus 8 \
    --collocate \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --critic_learning_rate 0.000001 \
    --lr_scheduler polynomial \
    --kl_penalty_coef 0 \
    --num_ppo_epochs 2 \
    --beta 0.0001 \
    --non_stop_penalty 0 \
    --oracle_type reward \
    --oracle gsm8k \
    --pretrain lkevinzc/rho-1b-sft-GSM8K \
    --apply-chat-template \
    --zero-stage 2 \
    --prompt_data lkevinzc/gsm8k \
    --max-train 9999999 \
    --num_prompt_epoch 6 \
    --prompt_max_length 1000 \
    --sync_params_every 1 \
    --num_samples 8 \
    --max_step_adjustment 16 \
    --critic_max_step_adjustment 16 \
    --temperature 1.2 \
    --top_p 0.9 \
    --generate_max_length 1024 \
    --save_steps -1 \
    --input_key question \
    --output_key final_answer \
    --train_split train \
    --train_batch_size 64 \
    --train_batch_size_per_device 8 \
    --mini_train_batch_size_per_device 8 \
    --rollout_batch_size 64 \
    --rollout_batch_size_per_device 8 \
    --pi_buffer_maxlen_per_device 64 \
    --max_eval 1500 \
    --eval_batch_size 200 \
    --eval_generate_max_length 1024 \
    --eval_steps 50 \
    --eval_temperature 0.35 \
    --eval_top_p 0.9 \
    --eval_n 16 \
    --eval_input_key question \
    --eval_output_key final_answer \
    --use-wb \
    --wb-run-name rho-1b-ppo-gsm8k
```
The experiment finishes in about 6 hours with 8 A100 GPUs.

With this, we ablated how *Boltzmann exploration* (controlled by the sampling temperature) affects the learning efficiency:

<p align="center">
  <img src="./ppo_temperature.png" width=55%/>
</p>

We look forward future studies on other efficient exploration strategies to enhance LLM reasoning.
