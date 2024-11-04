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
"""Custom RMs for tl;dr dataset built on Pythia."""
import os
from dataclasses import dataclass
from typing import List

import torch
import tyro
from mosec import Runtime, Server, Worker
from mosec.mixin import TypedMsgPackMixin
from msgspec import Struct
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl.trainer.utils import get_reward

from oat.utils.data import zero_pad_sequences


class Request(Struct, kw_only=True):
    batch_prompt: List[str]
    batch_candidates: List[List[str]]


class Response(Struct, kw_only=True):
    batch_first_win_prob: List[float]


example = Request(
    batch_prompt=[
        "What is the range of the numeric output of a sigmoid node in a neural network?"
    ]
    * 8,
    batch_candidates=[
        [
            "The output of a sigmoid node is bounded between -1 and 1.",
            "The output of a sigmoid node is bounded between 0 and 1.",
        ]
    ]
    * 8,
)


class PythiaCustomRewardModel(TypedMsgPackMixin, Worker):
    def __init__(self):
        super().__init__()
        model_name = os.environ.get("RM_MODEL_NAME")
        tokenizer_name = os.environ.get("TOKENIZER_NAME")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval().to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.example = example  # To warmup: do one forward pass to allocate GPU memory

    def forward(self, request: Request) -> Response:
        assert self.max_batch_size == 1
        prompts = self.tokenizer(
            request.batch_prompt, return_tensors="pt", padding=True
        )
        num_data, context_length = prompts["input_ids"].shape
        prompt_ids = prompts["input_ids"].repeat(2, 1)
        completion_ids = []
        for c in [c[0] for c in request.batch_candidates] + [
            c[1] for c in request.batch_candidates
        ]:
            completion_ids.append(
                self.tokenizer(
                    c,
                    return_tensors="pt",
                    padding=False,
                )["input_ids"]
            )
        completion_ids = zero_pad_sequences(
            completion_ids, side="right", value=self.tokenizer.pad_token_id
        ).squeeze()
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1).to(
            self.model.device
        )
        with torch.no_grad():
            _, logits, _ = get_reward(
                self.model,
                prompt_completion_ids,
                self.tokenizer.pad_token_id,
                context_length,
            )
        batch_scores_1 = logits[:num_data]
        batch_scores_2 = logits[num_data:]
        # Apply BT model.
        batch_first_win_prob = (batch_scores_1 - batch_scores_2).sigmoid().tolist()

        responses = Response(batch_first_win_prob=batch_first_win_prob)
        return responses


@dataclass
class ServerArgs:
    remote_rm_model: str = "trl-lib/pythia-1b-deduped-tldr-rm"
    tokenizer: str = ""
    max_wait_time: int = 10
    cuda_devices: str = "all"


if __name__ == "__main__":
    args = tyro.cli(ServerArgs)

    if args.tokenizer == "":
        args.tokenizer = args.remote_rm_model

    if args.cuda_devices == "all":
        NUM_DEVICE = torch.cuda.device_count()
        devices = list(range(NUM_DEVICE))
    else:
        devices = args.cuda_devices.split(",")
        NUM_DEVICE = len(devices)

    def _prepare_env(cid: int) -> dict:
        return {
            "CUDA_VISIBLE_DEVICES": str(cid),
            "RM_MODEL_NAME": args.remote_rm_model,
            "TOKENIZER_NAME": args.tokenizer,
        }

    server = Server()
    runtime = Runtime(
        worker=PythiaCustomRewardModel,
        num=NUM_DEVICE,
        max_batch_size=1,
        env=[_prepare_env(x) for x in devices],
        max_wait_time=args.max_wait_time,
        timeout=10,
    )
    server.register_runtime(
        {
            "/compare": [runtime],
        }
    )
    server.run()
