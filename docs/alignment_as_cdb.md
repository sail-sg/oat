## LLM alignment as contextual dueling bandits

LLM alignment is essentially an online learning and decision making problem where the **agent** (e.g., the LLM policy with an optional built-in reward model) interacts with the **environment** (i.e., humans) to achieve either of the two distinct objectives: minimizing cumulative regret in the *Explore & Exploit* setting or minimizing anytime regret in the *Best Arm Identification* setting.

In our [paper](https://arxiv.org/abs/2411.01493), we formalize LLM alignment as a **contextual dueling bandit (CDB)** problem (see illustration below) and propose a sample-efficient alignment approach based on Thompson sampling.

<p align="center">
  <img src="https://gist.githubusercontent.com/lkevinzc/98afee30a5141d7068a0b35a88901a31/raw/e0da719024bdc16fb4a993a8405e15cb0cf2b53a/interface.png" width=80%/>
</p>

The CDB framework necessitates an efficient online training system to validate the proposed method and compare it with other baselines. Oat 🌾 is developed as part of this research initiative.

Using the CDB framework, existing LLM alignment paradigms can be summarized as follows:

<p align="center">
  <img src="https://gist.githubusercontent.com/lkevinzc/98afee30a5141d7068a0b35a88901a31/raw/acbb25a20dd6c1e7619539b0fa449076ade2f873/compare.png" width=95%/>
</p>

For more details, please check out our [paper](https://arxiv.org/abs/2411.01493)!


## Examples
Below is an example to align a `1-B Pythia` SFT Model on the `tl;dr` dataset using `online SimPO` with `PairRM` as the preference oracle:

> [!WARNING]
> Aligning with `PairRM` provides a lightweight example setup. For reproducing results from the paper or developing custom online alignment algorithms, we recommend using stronger reward models (or GPT-as-a-judge) as a preference oracle. This approach better approximates the ideal case of a human population. See the [examples](./examples/README.md#preference-oracles).

```diff
python -m oat.experiment.main \
    --gpus 2 \
    --collocate \
    --dap-algo SimPO \
    --beta 2 \
    --preference-oracle pairrm \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --output_key pythia-1b-reference \
    --sync-params-every 1 \
    --rollout-batch-size-per-device 64 \
    --pi-buffer-maxlen-per-device 64 \
    --train-batch-size-per-device 8 \
    --use-wb \
    --wb-run-name 1b_pairrm_simpo_online
```
This example completes in **less than two hours on two A100-40G GPUs**!

To run an `offline SimPO` baseline for comparison, we disable weights synchronization from the learner to actors by adjusting the `sync-params-every` argument:
```diff
python -m oat.experiment.main \
    --gpus 2 \
    --collocate \
    --dap-algo SimPO \
    --beta 2 \
    --preference-oracle pairrm \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --output_key pythia-1b-reference \
-   --sync-params-every 1 \
+   --sync-params-every 9999 \ # any number > total gradient step (50000//128=390)
    --rollout-batch-size-per-device 64 \
    --pi-buffer-maxlen-per-device 64 \
    --train-batch-size-per-device 8 \
    --use-wb \
-   --wb-run-name 1b_pairrm_simpo_online
+   --wb-run-name 1b_pairrm_simpo_offline
```

Finally, we run `SEA SimPO` (with $\gamma=1$, see [here](https://arxiv.org/pdf/2411.01493#page=7.60) for the meaning of $\gamma$) to verify its capability of sample-efficient alignment. This experiment utilizes 4 GPUs, with a reduced per-device training batch size to accommodate the training of an additional epistemic reward model. The per-device rollout batch size and buffer length are adjusted to ensure a global batch size of 128. Additionally, 10 response candidates are generated for exploration using BAI Thompson sampling.
```diff
python -m oat.experiment.main \
-   --gpus 2 \
+   --gpus 4 \
    --dap-algo SimPO \
    --beta 2 \
    --preference-oracle pairrm \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --output_key pythia-1b-reference \
    --sync-params-every 1 \
-   --rollout-batch-size-per-device 64 \
-   --pi-buffer-maxlen-per-device 64 \
-   --train-batch-size-per-device 8 \
+   --rollout-batch-size-per-device 32 \
+   --pi-buffer-maxlen-per-device 32 \
+   --train-batch-size-per-device 1 \
+   --learn-rm \
+   --exp-method EnnBAITS \
+   --num_samples 10 \
    --use-wb \
-   --wb-run-name 1b_pairrm_simpo_online
+   --wb-run-name 1b_pairrm_simpo_sea
```

<p align="center">
  <img src="https://gist.githubusercontent.com/lkevinzc/98afee30a5141d7068a0b35a88901a31/raw/e23f40d33e8a2fa4220e8122c152b356084b8afb/example_result.png" width=55%/>
</p>

Check out this [tutorial](./preference_learning.md) for more examples covering:
* Various direct optimizers, including DPO, IPO, and SLiC.
* Different modes of preference oracles, such as remote reward models and GPT-as-a-judge.
* Additional LLM exploration algorithms, e.g., APL, XPO, and EE4LLM.
