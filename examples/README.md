This document provides extensive examples demonstrating how to use `oat` to (1) run various direct optimizers, (2) integrate different preference oracles, and (3) implement diverse active exploration algorithms. All the examples are tested on a machine with 8 A100 GPUs, with training logs publicly available on [wandb](https://wandb.ai/lkevinzc/oat-llm/) for reproducibility.


First of all, you could always check all supported arguments by running:
```terminal
python -m oat.experiment.main -h
```

### (1) running various direct optimizers

`oat` currently supports DPO, IPO, SLiC, and SimPO by setting `--dap-algo`. Remember to adjust the associated hyper-parameter `beta`.

```diff
python -m oat.experiment.main \
+   --dap-algo SLiC \
+   --beta 0.1 \
```

### (2) integrating different preference oracles
In the [main page](../README.md#usage) we have shown the usage of `pairrm` as the preference oracle, which runs locally in the same process as the actor. Next, we give an example of training `online DPO` with **a remote preference oracle served using [Mosec](https://github.com/mosecorg/mosec)**.

First, we start the Mosec service, which will serve 4 `Skywork/Skywork-Reward-Llama-3.1-8B` parallel workers as the preference oracle on the first 4 GPUs:
```terminal
MOSEC_LOG_LEVEL=debug python -m oat.oracles.remote.server --cuda-devices 0,1,2,3
```
After the service is up (seeing "http service is running" from the log), start a new bash and run:
```diff
python -m oat.experiment.main \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --dap-algo DPO \
    --beta 0.1 \
+   --reward-oracle remote \
+   --remote-rm-url http://0.0.0.0:8000 \
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
    --wb-run-name 1b_skywork_dpo_online
```

Alternatively, we could also query OpenAI API to use **GPT-as-a-judge as the preference oracle**:
```diff
python -m oat.experiment.main \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --dap-algo DPO \
    --beta 0.1 \
+   --reward_oracle gpt-4o-mini-2024-07-18 \
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
```

### (2) implement diverse active exploration algorithms
#### [SEA] Sample-Efficient Alignment for LLMs

We natively support SEA using the `main` entry script. For example, running `SEA DPO` with the remote preference oracle:
```diff
# Make sure the remote service is already available.

python -m oat.experiment.main \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --dap-algo DPO \
    --beta 0.1 \
    --reward-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
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
+   --num-prompt-epoch 2 \
+   --max-step-adjustment 0.75 \
+   --lr-warmup-ratio 0.02 \
+   --eval-query-interval 2560 \
+   --num-samples 20 \
+   --learn-rm \
+   --exp-method EnnBAITS \
+   --model-rollout \
+   --max-model-data-ratio 0.3 \
    --use-wb \
    --wb-run-name 1b_skywork_dpo_sea
```
