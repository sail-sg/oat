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

import logging
import math
import os
import random
from typing import Callable, List, Tuple

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from oat.types import PreferenceData
from oat.utils.deepspeed import DeepspeedStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrain, trust_remote_code=True, use_fast=use_fast
    )
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def load_data_from_disk_or_hf(data_name):
    if os.path.exists(data_name):
        return datasets.load_from_disk(data_name)
    return datasets.load_dataset(data_name)


def get_datasets(tokenizer, strategy, eval_only=False):
    args = strategy.args
    if not eval_only or args.eval_data == "":
        prompt_dataset = load_data_from_disk_or_hf(args.prompt_data)
        prompts_data = prompt_dataset[args.train_split].select(
            range(min(args.max_train, len(prompt_dataset[args.train_split])))
        )
    if args.eval_data:
        strategy.print(f"loading eval data {args.eval_data}")
        if "@" in args.eval_data:
            name, path = args.eval_data.split("@")
        else:
            name, path = None, args.eval_data
        eval_dataset = datasets.load_dataset(path, name, trust_remote_code=True)
    else:
        # Share the same dataset but use different split.
        eval_dataset = prompt_dataset

    eval_prompts_data = eval_dataset[args.eval_split].select(
        range(min(args.max_eval, len(eval_dataset[args.eval_split])))
    )
    if not eval_only:
        prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_key=args.input_key,
            output_key=args.output_key,
            apply_chat_template=args.apply_chat_template,
            get_reference=True,
        )
    else:
        prompts_dataset = None
    eval_prompts_dataset = PromptDataset(
        eval_prompts_data,
        tokenizer,
        strategy,
        input_key=args.eval_input_key or args.input_key,
        output_key=args.eval_output_key or args.output_key,
        apply_chat_template=args.apply_chat_template,
        get_reference=True,
    )
    return prompts_dataset, eval_prompts_dataset


def shard_buffer(
    dataset,
    rank: int,
    num_replicas: int,
    seed: int,
    shuffle=True,
    drop_last=True,
):
    if drop_last and len(dataset) % num_replicas != 0:
        # Ensure each rank receives the same amount of data.
        num_samples = math.ceil((len(dataset) - num_replicas) / num_replicas)
    else:
        num_samples = math.ceil(len(dataset) / num_replicas)
    total_size = num_samples * num_replicas
    indices = list(range(len(dataset)))
    if shuffle:
        # deterministically shuffle based on seed
        random.Random(seed).shuffle(indices)
    if not drop_last:
        padding_size = total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            dataset += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
    else:
        indices = indices[:total_size]
    assert len(indices) == total_size
    indices = indices[rank:total_size:num_replicas]
    assert len(indices) == num_samples
    return [dataset[i] for i in indices]


def pad_to_length(tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def _preprocess_preference_data(
    data: PreferenceData,
    apply_chat_template=None,
) -> Tuple[str, str, str, bool]:
    if apply_chat_template:
        prompt = {"content": data.prompt, "role": "user"}
        chosen = {"content": data.chosen_response, "role": "assistant"}
        rejected = {"content": data.rejected_response, "role": "assistant"}
        chosen = apply_chat_template([prompt, chosen], tokenize=False)
        rejected = apply_chat_template([prompt, rejected], tokenize=False)

        prompt = apply_chat_template(
            [prompt], tokenize=False, add_generation_prompt=True
        )
        chosen = chosen[len(prompt) :]
        rejected = rejected[len(prompt) :]
    else:
        prompt = data.prompt
        chosen = data.chosen_response
        rejected = data.rejected_response

    return prompt, chosen, rejected, data.same, data.chosen_id


class PromptDataset(Dataset):
    """Dataset for processing prompts."""

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_key,
        output_key=None,
        apply_chat_template=False,
        get_reference=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.get_reference = get_reference
        self.prompt_max_length = strategy.args.prompt_max_length

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
        if get_reference:
            assert output_key is not None

        self.raw_prompts = []
        self.processed_prompts = []
        self.references = []

        def preprocess_data(data, input_key="input", apply_chat_template=None) -> str:
            if apply_chat_template:
                prompt = apply_chat_template(
                    [{"content": data[input_key], "role": "user"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = data[input_key]
            if get_reference:
                return data[input_key], prompt, data[output_key]
            return data[input_key], prompt

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            if get_reference:
                prompt, processed_prompt, reference = preprocess_data(
                    data, input_key, apply_chat_template
                )
                self.references.append(reference)
            else:
                prompt, processed_prompt = preprocess_data(
                    data, input_key, apply_chat_template
                )
            if len(tokenizer(processed_prompt)["input_ids"]) <= self.prompt_max_length:
                self.processed_prompts.append(processed_prompt)
                self.raw_prompts.append(prompt)

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        if self.get_reference:
            return (
                self.processed_prompts[idx],
                self.raw_prompts[idx],
                self.references[idx],
            )
        return self.processed_prompts[idx], self.raw_prompts[idx]


class PreferenceDataset(Dataset):
    def __init__(
        self,
        buffer: List[PreferenceData],
        tokenizer: Callable,
        prompt_max_length: int,
        generate_max_length: int,
        strategy: DeepspeedStrategy,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.chosen_responses = []
        self.rejected_responses = []
        self.prompt_ids_lens = []
        self.same_masks = []
        self.chosen_ids = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.prompt_max_length = prompt_max_length
        self.generate_max_length = generate_max_length

        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            strategy.print("Applying chat template...")
            apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(
                self.strategy.args, "tokenizer_chat_template", None
            )
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        self.strategy.print("Constructing preference dataset...")

        for data in tqdm(buffer, disable=not self.strategy.is_rank_0()):
            prompt, chosen, rejected, same_mask, chosen_id = (
                _preprocess_preference_data(
                    data,
                    apply_chat_template,
                )
            )
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.prompt_max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.prompt_max_length - 2:
                logging.warn(
                    "Dropping samples due to length limit; this may cause the training hang because of synchronization"
                )
                continue
            else:
                self.prompt_ids_lens.append(prompt_ids_len)

            self.prompts.append(prompt)
            self.chosen_responses.append(chosen)
            self.rejected_responses.append(rejected)
            self.same_masks.append(same_mask)
            self.chosen_ids.append(chosen_id)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt, chosen, rejected = (
            self.prompts[idx],
            self.chosen_responses[idx],
            self.rejected_responses[idx],
        )
        extra = {
            "prompt_ids_lens": self.prompt_ids_lens[idx],
            "same_masks": self.same_masks[idx],
            "chosen_ids": self.chosen_ids[idx],
        }  # Modify collate_fn below as well.

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.prompt_max_length + self.generate_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        rejected = (prompt + rejected).rstrip("\n")
        if not rejected.endswith(self.tokenizer.eos_token):
            rejected += " " + self.tokenizer.eos_token
        rejected_token = self.tokenizer(
            rejected,
            max_length=self.prompt_max_length + self.generate_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        rejected_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        rejected_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            rejected_token["input_ids"],
            rejected_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        rejected_ids = []
        rejected_masks = []
        extras = {"prompt_ids_lens": [], "same_masks": [], "chosen_ids": []}
        for chosen_id, chosen_mask, rejected_id, rejected_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            rejected_ids.append(rejected_id)
            rejected_masks.append(rejected_mask)
            extras["prompt_ids_lens"].append(extra["prompt_ids_lens"])
            extras["same_masks"].append(extra["same_masks"])
            extras["chosen_ids"].append(extra["chosen_ids"])

        padding_side = "right"
        chosen_ids = zero_pad_sequences(
            chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id
        )
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        rejected_ids = zero_pad_sequences(
            rejected_ids, side=padding_side, value=self.tokenizer.pad_token_id
        )
        rejected_masks = zero_pad_sequences(rejected_masks, side=padding_side)
        return chosen_ids, chosen_masks, rejected_ids, rejected_masks, extras
