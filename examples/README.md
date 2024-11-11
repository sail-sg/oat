This document provides extensive examples demonstrating how to use oat 🌾 to (1) run various direct optimizers, (2) integrate different preference oracles, and (3) implement diverse active exploration algorithms. All the examples are tested on a machine with 8 A100 GPUs, with training logs publicly available on [wandb](https://wandb.ai/lkevinzc/oat-llm/) for reproducibility.

- [Direct optimizers](#direct-optimizers)
- [Preference oracles](#preference-oracles)
  - [Locally hosted Mosec service](#locally-hosted-mosec-service)
  - [OpenAI API for GPT-as-a-judge](#openai-api-for-gpt-as-a-judge)
  - [Scale up with remote Mosec service](#scale-up-with-remote-mosec-service)
- [LLM exploration algorithms](#llm-exploration-algorithms)
  - [\[SEA\] Sample-Efficient Alignment for LLMs](#sea-sample-efficient-alignment-for-llms)
  - [\[EE4LLM\] Efficient Exploration for LLMs](#ee4llm-efficient-exploration-for-llms)
  - [\[APL\] Active Preference Learning for Large Language Models](#apl-active-preference-learning-for-large-language-models)
  - [\[XPO\] Exploratory Preference Optimization](#xpo-exploratory-preference-optimization)


First of all, you could always check all supported arguments by running:
```terminal
python -m oat.experiment.main -h
```

## Direct optimizers

`oat` currently supports DPO, IPO, SLiC, and SimPO by setting `--dap-algo`. Remember to adjust the associated hyper-parameter `beta`.

```diff
python -m oat.experiment.main \
+   --dap-algo IPO \
+   --beta 0.1 \
    # other flags...
```

## Preference oracles
### Locally hosted Mosec service
In the [main page](../README.md#usage) we have shown the usage of `pairrm` as the preference oracle, which runs in the same process as the actor. Next, we give an example of training `online DPO` with **a preference oracle served using [Mosec](https://github.com/mosecorg/mosec)**.

First, we start the Mosec service locally, which will serve 4 `Skywork/Skywork-Reward-Llama-3.1-8B` parallel workers as the preference oracle on the first 4 GPUs:
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
+   --preference-oracle remote \
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

### OpenAI API for GPT-as-a-judge
Alternatively, we could also query OpenAI API to use **GPT-as-a-judge as the preference oracle**:
```diff
python -m oat.experiment.main \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
+   --collocate \
    --dap-algo DPO \
    --beta 0.1 \
+   --preference-oracle gpt-4o-mini-2024-07-18 \
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
We enabled collocation of learner and actor workers given the abundant GPU memory, thanks to the fact that the preference oracle (GPT) runs on OpenAI's side and almost takes no resource on our machines.

### Scale up with remote Mosec service
Likewise, we can also host our own remote server for any reward model *on a separate machine*, utilizing more compute to train larger models. With a Kubernetes-managed cluster, you could follow [these steps](../k8s/) to serve a remote preference oracle at `http://remote-rm`. Otherwise, you may need to obtain the remote machine's IP address (e.g., 10.0.0.1), and set `--remote-rm-url http://10.0.0.1:8000` accordingly.

```diff
python -m oat.experiment.main \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
+   --remote-rm-url http://remote-rm \
+   --pretrain trl-lib/pythia-6.9b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --input-key prompt \
+   --output-key pythia-6.9b-reference \
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
+   --wb-run-name 6.9b_skywork_dpo_online
```

## LLM exploration algorithms

All examples below assume a locally served preference oracle as done in the [section above](#locally-hosted-mosec-service).

### [SEA] Sample-Efficient Alignment for LLMs

> [!NOTE]
> Paper: https://arxiv.org/pdf/2411.01493.
> 
> You can find a thorough comparison between all algorithms mentioned in this section in our paper.

Oat natively supports SEA using the `oat.experiment.main` entry script:
```diff
python -m oat.experiment.main \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
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

### [EE4LLM] Efficient Exploration for LLMs

> [!NOTE]
> Paper: https://arxiv.org/pdf/2402.00396

Run EE4LLM by disabling policy learning and enabling best-of-n sampling for evaluation:
```diff
python -m oat.experiment.main \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
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
+   --num-samples 20 \
+   --learn-rm \
+   --learn_rm_only \
+   --exp-method EnnEETS \
+   --exp_rnd_sample \
+   --online_evaluation \
+   --best_of_n_eval \
+   --num_bon 10 \
    --use-wb \
+   --wb-run-name 1b_skywork_dpo_ee4llm
```

### [APL] Active Preference Learning for Large Language Models

> [!NOTE]
> Paper: https://arxiv.org/pdf/2402.08114

APL can be implemented by inheriting oat's learner and actor classes ([codes](../oat/baselines/apl.py)). Run it with a dedicated entry script:
```diff
+ python -m oat.experiment.run_apl \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
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
+   --num_prompt_epoch 4 \
+   --max_train 100000 \
+   --max_step_adjustment 0.125 \
+   --num_samples 8 \
+   --apl_pref_certainty_only \
    --use-wb \
+   --wb-run-name 1b_skywork_apl
```

### [XPO] Exploratory Preference Optimization

> [!NOTE]
> Paper: https://arxiv.org/pdf/2405.21046

XPO can be implemented by inheriting oat's learner and actor classes ([codes](../oat/baselines/xpo.py)). Run it with a dedicated entry script:
```diff
+ python -m oat.experiment.run_xpo \
    --flash-attn \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
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
    --use-wb \
+   --wb-run-name 1b_skywork_xpo
```
