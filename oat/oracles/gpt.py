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
import os
import threading
import time
import traceback
from typing import Any, List, Sequence, Tuple
from warnings import warn

import numpy as np
from openai import OpenAI
from scipy.special import logsumexp

from oat.oracles.base import PreferenceOracleBase
from oat.types import Metric

DEFAULT_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''


class GPTJudgeOracle(PreferenceOracleBase):
    def __init__(
        self,
        reward_model_path: str,
        shuffle_order: bool = True,
        max_workers: int = 4,
        max_retry: int = 10,
        **_,
    ) -> None:
        super().__init__()
        self.client = OpenAI(
            api_key=os.environ.get("OAI_KEY"),
            base_url=os.environ.get("OAI_URL"),
        )
        self.model = reward_model_path
        self.shuffle_order = shuffle_order
        self.invalid_count = 0
        self.max_workers = max_workers
        self.max_retry = max_retry
        self.mutex = threading.Lock()
        self.template = DEFAULT_PAIRWISE_SYSTEM_PROMPT

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        del batch_size, disable_tqdm

        completions = list(zip(candidates_A, candidates_B))

        # Shuffle the order of the completions to avoid positional bias
        if self.shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(inputs))
            completions = [
                pair[::-1] if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.template.format(
                prompt=prompt, response0=candidates[0], response1=candidates[1]
            )
            messages = [{"role": "user", "content": content}]

            wait_time = 1
            for _ in range(self.max_retry):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1,
                        logprobs=True,
                        top_logprobs=5,
                        temperature=0,
                    )
                    break
                except Exception as e:
                    warn(f"OpenAI API failure: {e} {traceback.format_exc()}")
                    time.sleep(wait_time)
                    wait_time *= 1.3
            else:
                raise RuntimeError("OpenAI API error!")

            first_win_prob = logprob_parser(
                completion, numerator_token="0", denominator_tokens=["0", "1"]
            )
            if np.isnan(first_win_prob):
                logging.warn("Invalid win prob!")
                with self.mutex:
                    self.invalid_count += 1
                return np.random.uniform(0, 1)
            else:
                return first_win_prob

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            first_win_probs = list(executor.map(get_rank, inputs, completions))

        # Flip back the ranks to the original order if needed
        if self.shuffle_order:
            first_win_probs = [
                first_win_probs[i] if not flip else 1 - first_win_probs[i]
                for i, flip in enumerate(flip_mask)
            ]
        first_win_probs = np.array(first_win_probs)
        if return_probs:
            return first_win_probs, {}
        else:
            return first_win_probs > 0.5, {}


def logprob_parser(
    completion: dict,
    numerator_token: str,
    denominator_tokens: Sequence[str],
) -> float:
    top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
    map_tokens_to_logprobs = {
        t.token: t.logprob
        for t in top_logprobs
        if t.token in denominator_tokens + [numerator_token]
    }
    missing = float("-inf")
    if len(map_tokens_to_logprobs) == 0:
        return np.nan

    baseline_logprob = map_tokens_to_logprobs.get(numerator_token, missing)
    denominator_logprob = logsumexp(
        [map_tokens_to_logprobs.get(t, missing) for t in denominator_tokens]
    )

    out_logprob = baseline_logprob - denominator_logprob
    probability = np.exp(out_logprob)
    return probability
