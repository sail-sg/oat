<p align="center">
  <img src="https://private-user-images.githubusercontent.com/38581401/380485232-6fa9045a-8f70-495b-bdec-2e4664044c41.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzA3NzA3NjEsIm5iZiI6MTczMDc3MDQ2MSwicGF0aCI6Ii8zODU4MTQwMS8zODA0ODUyMzItNmZhOTA0NWEtOGY3MC00OTViLWJkZWMtMmU0NjY0MDQ0YzQxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTA1VDAxMzQyMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI4NTEzYjVhMzY2YWI5MjZlMjhkNGVkYjcwN2M1NTY1Y2Y2M2UyYjY3Mjc1MTYzYmYzZmQwYThmMWU0YTc2MjkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.cn8rX_LZ-LKD1Z1zA_JBvYZJyVWIN4S9zWleT3XZu9Q" height="230" alt="OAT" />
</p>

[![PyPI - Version](https://img.shields.io/pypi/v/oat-llm.svg)](https://pypi.org/project/oat-llm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oat-llm.svg)](https://pypi.org/project/oat-llm)
<a href="">
<img src="https://img.shields.io/github/license/sail-sg/oat" alt="License">
<a href="https://arxiv.org/pdf/2411.01493">
<img src="https://img.shields.io/badge/arXiv-2411.01493-b31b1b.svg" alt="Arxiv">

## Introduction

Oat 🌾 is a simple yet efficient system for running online LLM alignment algorithms. Key features include:

* **High Efficiency**: Oat implements a distributed *Actor-Learner-Oracle* architecture, with each component being optimized using state-of-the-art tools:
  * `Actor`: Powered by [vLLM](https://github.com/vllm-project/vllm) for accelerated online response sampling.
  * `Learner`: Enhanced by [DeepSpeed](https://github.com/microsoft/DeepSpeed) ZeRO strategies to maximize memory efficiency.
  * `Oracle`: Hosted by [Mosec](https://github.com/mosecorg/mosec) as a remote service, supporting dynamic batching, data parallelism and pipeline parallelism.
* **Simplification**: Oat simplifies the experimental pipeline of LLM alignment. With an `Oracle` served online, we can flexibly query it for preference data labeling as well as anytime model evaluation. All you need is to launch experiments and monitor real-time learning curves (e.g., win rate) on [wandb](https://wandb.ai/lkevinzc/oat-llm) — no need for manual training, checkpointing and loading for evaluation.
* **Oracle Simulation**: Oat provides simulated preference oracles in various modes.
  * Lightweight reward models can run directly in the actor's process, facilitating fast testing with as few as two GPUs.
  * Larger and more capable reward models can be served remotely, harnessing additional compute and memory resources.
  * LLM-as-a-judge is supported via querying OpenAI API for model-based pairwise ranking.
* **Ease of Use**: With a modular design, oat allows researchers to inherit and modify existing classes effortlessly, enabling rapid prototyping of algorithms.
* **Cutting-Edge Algorithms**: Oat implements state-of-the-art LLM exploration (active alignment) algorithms, including SEA, APL and XPO, along with popular direct optimizers such as DPO and SimPO, fostering innovation and fair benchmarking.

## Sample-efficient alignment for LLMs
If LLM leads us to AGI, it must be an online learner (see [Rich Sutton's thought on learning](https://www.youtube.com/watch?v=NvfK1TkXmOQ)). Online alignment is a crucial part of LLM's online learning process, where the LLM agent continually interacts with human beings and aligns with humans' values. 

## Installation
In a python environment with supported versions (`>=3.8, <=3.10`), you could install oat via PyPI:
```console
pip install vllm==0.6.2 && pip install oat-llm
```
Or you could also install in "editable" mode for local development:
```console
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

To run an `offline SimPO` for comparison, we disable weights synchronization from the learner to actors by adjusting the `sync-params-every` argument:
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

Finally, we run `SEA SimPO` (with $\gamma=1$) to verify its capability of sample-efficient alignment. We now utilize 4 GPUs and reduce per-device training batch size to accommodate learning an additional epistemic reward model. We also adjust per-device rollout batch size and buffer length accordingly to make the global size 128. Besides, 10 response candidates are generated for exploration by BAI Thompson sampling.
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
  <img src="https://private-user-images.githubusercontent.com/38581401/382972481-590b5318-56a5-4103-b937-1f05ffb90ebc.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzA3NzA3NjEsIm5iZiI6MTczMDc3MDQ2MSwicGF0aCI6Ii8zODU4MTQwMS8zODI5NzI0ODEtNTkwYjUzMTgtNTZhNS00MTAzLWI5MzctMWYwNWZmYjkwZWJjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTA1VDAxMzQyMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTY1YzRkYTI1MGZiMzhkOTUxMDVkZDRiODBlYjYwMGJkYjIxOTA0OTlkYjRhZGZjMGM3M2E2YzYyYzA1ZWM3NjUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.myLWPt2qvHyrdvaQP1A57J-BTvgLt0TdY8BhPYIugNA" height="330"/>
</p>

Check out this [tutorial](./examples/) for more examples regarding:
* Different direct optimizers including DPO, IPO, and SLiC.
* Various modes of preference oracles such as remote reward models and GPT-as-a-judge.
* Other LLM exploration algorithms, e.g., APL, XPO, and EE4LLM.

## Benchmarking
Our system benchmarking compares oat with the online DPO implementation from [huggingface/trl](https://huggingface.co/docs/trl/main/en/online_dpo_trainer). We use the following configurations for oat, and show the benchmarking results below.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/38581401/382674288-95021ec7-b79e-47f8-8f46-32f2f4212a36.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzA3NzA3NjEsIm5iZiI6MTczMDc3MDQ2MSwicGF0aCI6Ii8zODU4MTQwMS8zODI2NzQyODgtOTUwMjFlYzctYjc5ZS00N2Y4LThmNDYtMzJmMmY0MjEyYTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTA1VDAxMzQyMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTFjYzI2M2I2ZTUwYzUzMDA3YmViNzM5MjFkMjUwZGQyYjMyZDc1ZDAwNzAzMzNjNjJiN2EyNzQ3OGNjNmVmNWImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.bcBfZA2ZQu0naU7j36xfs5392CastXoKDODR3iARjqM" height="320"/>
</p>

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/38581401/382674313-2d145308-2983-4ee2-9413-61c083933327.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzA3NzA3NjEsIm5iZiI6MTczMDc3MDQ2MSwicGF0aCI6Ii8zODU4MTQwMS8zODI2NzQzMTMtMmQxNDUzMDgtMjk4My00ZWUyLTk0MTMtNjFjMDgzOTMzMzI3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTA1VDAxMzQyMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM5NmM4MDRhNjg4YjVhN2JmZTFlNmU1YmQ3NzZhNjc1MDNkNzBkYzIzMDMwNDg1ZTI5NmUxZTRlMmM3Zjc1ZWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ljiO86Q6Pl3cgEK04PlKD3AN2VDV1UrXoTyx3vXFUdo" height="300" alt="OAT" />
</p>

Please refer to the Appendix C of our paper for detailed benchmarking methods and discussion on the results.

## License

`oat` is distributed under the terms of the [Apache2](https://www.apache.org/licenses/LICENSE-2.0) license.

## Acknowledgement
We thank the following awesome projects which oat benefits from:
* [vLLM](https://github.com/vllm-project/vllm)
* [DeepSpeed](https://github.com/microsoft/DeepSpeed)
* [Mosec](https://github.com/mosecorg/mosec)
* [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

## Disclaimer

This is not an official Sea Limited or Garena Online Private Limited product.