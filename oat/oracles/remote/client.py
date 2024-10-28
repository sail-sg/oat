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

import concurrent.futures
import logging
import threading
import time
from http import HTTPStatus
from typing import Any, List
from warnings import warn

import httpx
import msgspec
import numpy as np
import tree

from oat.oracles.base import OracleBase

logging.getLogger("httpx").setLevel(logging.WARNING)


class RemoteRMOracle(OracleBase):
    def __init__(
        self,
        remote_rm_url: str,
        max_workers: int = 4,
        max_retry: int = 10,
        **_,
    ) -> None:
        super().__init__()
        self.client = httpx.Client(
            base_url=remote_rm_url,
        )
        self.invalid_count = 0
        self.max_workers = max_workers
        self.max_retry = max_retry
        self.mutex = threading.Lock()

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> List[Any]:
        del disable_tqdm

        completions = list(zip(candidates_A, candidates_B))

        batched_prompts = []
        batched_completions = []
        num = len(inputs)
        for ndx in range(0, num, batch_size):
            batched_prompts.append(inputs[ndx : min(ndx + batch_size, num)])
            batched_completions.append(completions[ndx : min(ndx + batch_size, num)])

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(bp, bc):
            wait_time = 1
            for _ in range(self.max_retry):
                try:
                    resp = self.client.post(
                        "/compare",
                        content=msgspec.msgpack.encode(
                            {
                                "batch_prompt": bp,
                                "batch_candidates": bc,
                            }
                        ),
                    )
                    if resp.status_code == HTTPStatus.OK:
                        result = msgspec.msgpack.decode(resp.content)
                        batch_first_win_prob = result["batch_first_win_prob"]
                        break
                    else:
                        raise RuntimeError(f"{resp.status_code}, {resp.content}")
                except Exception as e:
                    warn(f"Remote RM server failure: {e}")
                    time.sleep(wait_time)
                    wait_time *= 1.3
            else:
                raise RuntimeError("Remote RM server error!")

            return batch_first_win_prob

        # Call the remote server concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            batch_first_win_prob = list(
                executor.map(get_rank, batched_prompts, batched_completions)
            )

        first_win_probs = np.array(tree.flatten(batch_first_win_prob))
        if return_probs:
            return first_win_probs
        else:
            return first_win_probs > 0.5
