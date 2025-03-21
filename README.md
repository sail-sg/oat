<p align="center">
  <img src="./docs/new_logo.png" width=90% alt="OAT" />
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/oat-llm.svg)](https://pypi.org/project/oat-llm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oat-llm.svg)](https://pypi.org/project/oat-llm)
[![License](https://img.shields.io/github/license/sail-sg/oat)](https://github.com/sail-sg/oat/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2411.01493-b31b1b.svg)](https://arxiv.org/abs/2411.01493)

[Installation](#installation) | [Usage](#usage) | [Examples](./examples/) | [Benchmarking](#benchmarking) | [Citation](#citation)

---

## Updates
* 26/01/2025: We support reinforcement learning with verifiable rewards (RLVR) for math reasoning.
  * A quick [example](https://github.com/sail-sg/oat/blob/main/docs/reasoning_examples.md#deepseek-r1-zero-like-training) of R1-Zero-like training with GRPO.

## Introduction

Oat 🌾 is a simple yet efficient framework for running **online** LLM alignment algorithms. Its key features include:

* **High Efficiency**: Oat implements a distributed *Actor-Learner-Oracle* architecture, with each component being optimized using state-of-the-art tools:
  * `Actor`: Utilizes [vLLM](https://github.com/vllm-project/vllm) for accelerated online response sampling.
  * `Learner`: Leverages [DeepSpeed](https://github.com/microsoft/DeepSpeed) ZeRO strategies to enhance memory efficiency.
  * `Oracle`: Model-based oracle by [Mosec](https://github.com/mosecorg/mosec) as a remote service, supporting dynamic batching, data parallelism and pipeline parallelism.
* **Simplified Workflow**: Oat simplifies the experimental pipeline of LLM alignment. With an `Oracle` served online, we can flexibly query it for preference data labeling as well as anytime model evaluation. All you need is to launch experiments and monitor real-time learning curves (e.g., win rate) on wandb (see [reproduced results](https://wandb.ai/lkevinzc/oat-llm)) — no need for manual training, checkpointing and loading for evaluation.
* **Oracle Simulation**: Oat provides a diverse set of oracles to simulate preference/reward/verification feedback.
  * Verifiable rewards supported using rule-based functions.
  * Lightweight reward models run within the actor's process, enabling quick testing on as few as two GPUs.
  * Larger and more capable reward models can be served remotely, harnessing additional compute and memory resources.
  * LLM-as-a-judge is supported via querying OpenAI API for model-based pairwise ranking.
* **Ease of Use**: Oat's modular structure allows researchers to easily inherit and modify existing classes, enabling rapid prototyping and experimentation with new algorithms.
* **Cutting-Edge Algorithms**: Oat implements state-of-the-art online algorithms, fostering innovation and fair benchmarking.
  * PPO (online RL) for math reasoning.
  * Online DPO/SimPO/IPO for online preference learning.
  * Online exploration (active alignment) algorithms, including [SEA](https://arxiv.org/abs/2411.01493), APL and XPO.

## Installation
In a python environment with supported versions (`>=3.8, <=3.10`), you could install oat via PyPI:
```shell
pip install vllm==0.7.2 && pip install oat-llm
```
Or you could also install in "editable" mode for local development:
```shell
git clone git@github.com:sail-sg/oat.git
cd oat
pip install vllm==0.7.2 && pip install -e .
```

##  Usage
* [Improving math reasoning with PPO](./docs/reasoning_examples.md).
* [Online preference learning with active exploration](./docs/alignment_as_cdb.md).

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
If you find this codebase useful for your research, please consider citing
```
@misc{liu2025oat,
author       = {Zichen Liu and Changyu Chen and Chao Du and Wee Sun Lee and Min Lin},
title        = {OAT: A research-friendly framework for LLM online alignment},
howpublished = {[https://github.com/sail-sg/oat](https://github.com/sail-sg/oat)},
year         = {2025}
}
```

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
