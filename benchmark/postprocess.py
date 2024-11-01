# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tqdm
import wandb
from fire import Fire


def main(run_name: str, wandb_proj: str = "lkevinzc/oat-benchmark"):
    features_of_interest = [
        "actor/oracle_time",
        "actor/generate_time",
        "misc/gradient_update_elapse",
        "train/learn_batch_time",
    ]

    api = wandb.Api()
    runs = api.runs(wandb_proj)
    for run in tqdm.tqdm(runs):
        if run.name == run_name:
            print(run.name)
            data = run.history(keys=features_of_interest)
            break

    print(data)
    print(data.iloc[range(5, 15), :].mean())


if __name__ == "__main__":
    Fire(main)
