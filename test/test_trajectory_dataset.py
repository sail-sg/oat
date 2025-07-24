# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test multi-turn trajectory dataset."""
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from oat.types import TrajectoryData
from oat.utils.data import TrajectoryDataset
from oat.utils.deepspeed import DummyStrategy


def test():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    dataset = load_dataset("robinsmits/ChatAlpaca-20K")["test"]

    ds = TrajectoryDataset(
        [
            TrajectoryData(
                trajectory_ids=None,
                num_turns=None,
                response_token_ranges=None,
                turn_weights=None,
                messages=x,
            )
            for x in dataset["messages"]
        ],
        tokenizer,
        DummyStrategy(None),
    )

    dl = DataLoader(
        ds,
        batch_size=len(ds),
        shuffle=(False),
        drop_last=True,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )

    input_ids, attention_mask, turn_weights, response_token_ranges = next(iter(dl))
    del attention_mask, turn_weights
    idx = 0
    for i, (st, end) in enumerate(response_token_ranges[idx]):
        print(f"turn-{i}", tokenizer.decode(input_ids[idx][st:end]))


if __name__ == "__main__":
    test()
