<p align="center">
  <img src="./assets/logo.png" height="230" alt="OAT" />
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/oat.svg)](https://pypi.org/project/oat-llm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oat-llm.svg)](https://pypi.org/project/oat-llm)

## Introduction
Oat 🌾 is a simple yet efficient learning system for executing online LLM alignment algorithms. Its features include:

* **Highly efficient**: Oat implements a distributed *Actor-Learner-Oracle* architecture, with each component being optimized with state-of-the-art technologies:
  * Actor is built with [vLLM](https://github.com/vllm-project/vllm) to accelerate the online response sampling.
  * Learner utilizes [DeepSpeed](https://github.com/microsoft/DeepSpeed) ZeRO strategies for enhancing memory-efficiency.
  * Oracle is hosted by [Mosec](https://github.com/mosecorg/mosec) as a remote service that supports dynamic request batching, data parallel & pipeline parallel computation.
* **Easy-to-use**: With the modular Actor-Learner-Oracle design, researchers could simply inherit existing classes and make effortless modifications on any of the components to verify new algorithms. The experimentation pipeline is extremly simplified:
  * No more model saving, checkpointing and offline evaluation, etc. Thanks to the hosted reward Oracle, we can evaluate the model anytime and log the win rate to wandb!
## Installation :wrench:
Oat requires a python environment with `python==3.10`.
```console
pip install vllm==0.6.2 && pip install oat-llm
```
Or you could also install in "editable" mode for local development:
```console
git clone git@github.com:sail-sg/oat.git
cd oat
pip install vllm==0.6.2 && pip install -e .
```

## License

`oat` is distributed under the terms of the [Apache2](https://www.apache.org/licenses/LICENSE-2.0) license.
