<p align="center">
  <img src="https://gist.githubusercontent.com/lkevinzc/98afee30a5141d7068a0b35a88901a31/raw/e23f40d33e8a2fa4220e8122c152b356084b8afb/logo.png" width=90% alt="OAT" />
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/oat-llm.svg)](https://pypi.org/project/oat-llm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oat-llm.svg)](https://pypi.org/project/oat-llm)
[![License](https://img.shields.io/github/license/sail-sg/oat)](https://github.com/sail-sg/oat/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2411.01493-b31b1b.svg)](https://arxiv.org/abs/2411.01493)

[Installation](#installation) | [Usage](#usage) | [Examples](./examples/) | [Benchmarking](#benchmarking) | [Citation](#citation)

---

## Introduction

Oat 🌾 is a simple yet efficient system for running online LLM alignment algorithms. Its key features include:

* **High Efficiency**: Oat implements a distributed *Actor-Learner-Oracle* architecture, with each component being optimized using state-of-the-art tools:
  * `Actor`: Utilizes [vLLM](https://github.com/vllm-project/vllm) for accelerated online response sampling.
  * `Learner`: Leverages [DeepSpeed](https://github.com/microsoft/DeepSpeed) ZeRO strategies to enhance memory efficiency.
  * `Oracle`: Hosted by [Mosec](https://github.com/mosecorg/mosec) as a remote service, supporting dynamic batching, data parallelism and pipeline parallelism.
* **Simplified Workflow**: Oat simplifies the experimental pipeline of LLM alignment. With an `Oracle` served online, we can flexibly query it for preference data labeling as well as anytime model evaluation. All you need is to launch experiments and monitor real-time learning curves (e.g., win rate) on [wandb](https://wandb.ai/lkevinzc/oat-llm) — no need for manual training, checkpointing and loading for evaluation.
* **Oracle Simulation**: Oat provides simulated preference oracles in various modes.
  * Lightweight reward models run within the actor's process, enabling quick testing on as few as two GPUs.
  * Larger and more capable reward models can be served remotely, harnessing additional compute and memory resources.
  * LLM-as-a-judge is supported via querying OpenAI API for model-based pairwise ranking.
* **Ease of Use**: Oat's modular structure allows researchers to easily inherit and modify existing classes, enabling rapid prototyping and experimentation with new algorithms.
* **Cutting-Edge Algorithms**: Oat implements state-of-the-art LLM exploration (active alignment) algorithms, including SEA, APL and XPO, along with popular direct optimizers such as DPO and SimPO, fostering innovation and fair benchmarking.

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

## Installation
In a python environment with supported versions (`>=3.8, <=3.10`), you could install oat via PyPI:
```shell
pip install vllm==0.6.2 && pip install oat-llm
```
Or you could also install in "editable" mode for local development:
```shell
git clone git@github.com:sail-sg/oat.git
cd oat
pip install vllm==0.6.2 && pip install -e .
```

## Usage
Below is an example to align a `1-B Pythia` SFT Model on the `tl;dr` dataset using `online SimPO` with `PairRM` as the preference oracle:
```diff
python -m oat.experiment.main \
    --gpus 2 \
    --collocate \
    --dap-algo SimPO \
    --beta 2 \
    --reward-oracle pairrm \
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
This example completes in **less than two hours on two A100 GPUs**!

To run an `offline SimPO` baseline for comparison, we disable weights synchronization from the learner to actors by adjusting the `sync-params-every` argument:
```diff
python -m oat.experiment.main \
    --gpus 2 \
    --collocate \
    --dap-algo SimPO \
    --beta 2 \
    --reward-oracle pairrm \
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

Finally, we run `SEA SimPO` (with $\gamma=1$) to verify its capability of sample-efficient alignment. This experiment utilizes 4 GPUs, with a reduced per-device training batch size to accommodate the training of an additional epistemic reward model. The per-device rollout batch size and buffer length are adjusted to ensure a global batch size of 128. Additionally, 10 response candidates are generated for exploration using BAI Thompson sampling.
```diff
python -m oat.experiment.main \
-   --gpus 2 \
+   --gpus 4 \
    --dap-algo SimPO \
    --beta 2 \
    --reward-oracle pairrm \
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

Check out this [tutorial](./examples/) for more examples covering:
* Various direct optimizers, including DPO, IPO, and SLiC.
* Different modes of preference oracles, such as remote reward models and GPT-as-a-judge.
* Additional LLM exploration algorithms, e.g., APL, XPO, and EE4LLM.

## Benchmarking
The benchmarking compares oat with the online DPO implementation from [huggingface/trl](https://huggingface.co/docs/trl/main/en/online_dpo_trainer). Below, we outline the configurations used for oat and present the benchmarking results. Notably, oat 🌾 achieves up to **2.5x** computational efficiency compared to trl 🤗.

<p align="center">
  <img src="https://gist.githubusercontent.com/lkevinzc/98afee30a5141d7068a0b35a88901a31/raw/e23f40d33e8a2fa4220e8122c152b356084b8afb/system_configs.png" width=97%/>
</p>

<p align="center">
  <img src="https://gist.githubusercontent.com/lkevinzc/98afee30a5141d7068a0b35a88901a31/raw/e23f40d33e8a2fa4220e8122c152b356084b8afb/bench_results.png" width=65% />
</p>

Please refer to [Appendix C of our paper](https://arxiv.org/pdf/2411.01493#page=17.64) for a detailed discussion of the benchmarking methods and results.

## Citation
If you find this work useful for your research, please consider citing
```
@article{
  liu2024sea,
  title={Sample-Efficient Alignment for LLMs},
  author={Zichen Liu and Changyu Chen and Chao Du and Wee Sun Lee and Min Lin},
  journal={arXiv preprint arXiv:2411.01493},
  year={2024}
}
```

## License

`oat` is distributed under the terms of the [Apache2](https://www.apache.org/licenses/LICENSE-2.0) license.

## Acknowledgement
We thank the following awesome projects that have contributed to the development of oat:
* [vLLM](https://github.com/vllm-project/vllm)
* [DeepSpeed](https://github.com/microsoft/DeepSpeed)
* [Mosec](https://github.com/mosecorg/mosec)
* [launchpad](https://github.com/google-deepmind/launchpad)
* [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

## Disclaimer

This is not an official Sea Limited or Garena Online Private Limited product.
