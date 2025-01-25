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

import os
from dataclasses import dataclass
from typing import List

import torch
import tyro
from mosec import Runtime, Server, Worker
from mosec.mixin import TypedMsgPackMixin
from msgspec import Struct
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Request(Struct, kw_only=True):
    batch_prompt: List[str]
    batch_response: List[str]  # For reward oracle.
    batch_candidates: List[List[str]]  # For preference oracle.


class Response(Struct, kw_only=True):
    batch_score: List[float]  # For reward oracle.
    batch_first_win_prob: List[float]  # For preference oracle.


MODEL_CONFIGS = {
    "Skywork/Skywork-Reward-Llama-3.1-8B": {
        "attn_implementation": "flash_attention_2",
        "num_labels": 1,
    },
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2": {
        "attn_implementation": "flash_attention_2",
        "num_labels": 1,
    },
    "Skywork/Skywork-Reward-Gemma-2-27B-v0.2": {
        "attn_implementation": "flash_attention_2",
        "num_labels": 1,
    },
}


class RewardModel(TypedMsgPackMixin, Worker):
    def __init__(self):
        super().__init__()
        self.model_name = os.environ.get("RM_MODEL_NAME")
        configs = MODEL_CONFIGS.get(self.model_name, {})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **configs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.example = Request(
            batch_prompt=[
                "What is the range of the numeric output of a sigmoid node in a neural network?"
            ],
            batch_response=[],
            batch_candidates=[
                [
                    "The output of a sigmoid node is bounded between -1 and 1.",
                    "The output of a sigmoid node is bounded between 0 and 1.",
                ]
            ],
        )  # To warmup: do one forward pass to allocate GPU memory

    def forward(self, request: Request) -> Response:
        assert self.max_batch_size == 1

        batch_msg = []
        if request.batch_candidates:
            # Rank two candidates.
            batch_msg1 = []
            batch_msg2 = []
            num_data = len(request.batch_prompt)
            for i, prompt in enumerate(request.batch_prompt):
                msg1 = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": request.batch_candidates[i][0]},
                ]
                msg2 = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": request.batch_candidates[i][1]},
                ]
                batch_msg1.append(msg1)
                batch_msg2.append(msg2)
            batch_msg = batch_msg1 + batch_msg2
        elif request.batch_response:
            # Score a given response.
            for i, prompt in enumerate(request.batch_prompt):
                msg = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": request.batch_response[i]},
                ]
                batch_msg.append(msg)

        pair = self.tokenizer.apply_chat_template(batch_msg, tokenize=False)
        pair = self.tokenizer(pair, return_tensors="pt", padding=True).to(
            self.model.device
        )
        with torch.no_grad():
            logits = self.model(**pair).logits.cpu().float().squeeze()

        if request.batch_candidates:
            batch_scores_1 = logits[:num_data]
            batch_scores_2 = logits[num_data:]
            # Apply BT model.
            batch_first_win_prob = (batch_scores_1 - batch_scores_2).sigmoid().tolist()
            batch_score = []
        elif request.batch_response:
            batch_first_win_prob = []
            batch_score = logits.tolist()
        return Response(
            batch_score=batch_score, batch_first_win_prob=batch_first_win_prob
        )


@dataclass
class ServerArgs:
    remote_rm_model: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    max_wait_time: int = 10
    cuda_devices: str = "all"
    multi_gpu: bool = False


if __name__ == "__main__":
    args = tyro.cli(ServerArgs)

    if args.multi_gpu:
        NUM_DEVICE = 1
        devices = [",".join([str(i) for i in range(torch.cuda.device_count())])]
    else:
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
        }

    server = Server()
    runtime = Runtime(
        worker=RewardModel,
        num=NUM_DEVICE,
        max_batch_size=1,
        env=[_prepare_env(x) for x in devices],
        max_wait_time=args.max_wait_time,
        timeout=10,
    )
    server.register_runtime(
        {
            "/get_feedback": [runtime],
        }
    )
    server.run()
