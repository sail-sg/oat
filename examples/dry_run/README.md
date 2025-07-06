### Dry run debugging
We allow users to enable the "dry run" mode to test if the training will go OOM with a specified context length. It works by replacing the real data with dummy data with desired lengths in the dataset's `__getitem__` method.

Currently we support dry run for both SFT and RL, please see example commands in this directory. They have been well-tested on A100-40G GPUs.

* Set `--dry_run` flag to enable this mode.
* `--dry_run_prompt_len` specifies the input length.
* `--dry_run_response_len` specifies the output length.
