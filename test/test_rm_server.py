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

import time

from fire import Fire

from oat.oracles.remote.client import RemoteRMOracle


def main(batch_size: int = 8, max_workers: int = 4, server_addr: str = "0.0.0.0:8000"):
    remote_oracle = RemoteRMOracle(
        remote_rm_url=f"http://{server_addr}", max_workers=max_workers
    )
    prompts = [
        "What is the range of the numeric output of a sigmoid node in a neural network?"
    ]
    candidate_1 = ["The output of a tanh node is bounded between -1 and 1."]
    candidate_2 = ["The output of a sigmoid node is bounded between 0 and 1."]

    st = time.time()
    scores = remote_oracle.compare(
        prompts, candidate_1, candidate_2, batch_size=batch_size, return_probs=True
    )
    print(f"result: {scores}; time: {time.time() - st}")


if __name__ == "__main__":
    Fire(main)
