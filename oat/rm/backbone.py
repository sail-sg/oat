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

import abc
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    SequenceClassifierOutput,
)


def get_cls(model_name: str):
    if "pairrm" in model_name.lower():
        return DebertaV2PairRM
    if "deberta" in model_name.lower():
        return DebertaV2Vanilla
    if "skywork" in model_name.lower():
        return SkyWorkRM
    return PythiaPretrained


class RMBackbone(abc.ABC):
    tokenizer: AutoTokenizer
    source_prefix: str
    cand_prefix: str

    def tokenize_pair(
        self, prompt: str, candidate: str, source_max_length: int, max_length: int
    ):
        source_ids = self.tokenizer.encode(
            self.source_prefix + prompt,
            max_length=source_max_length,
            truncation=True,
        )
        candidate_max_length = max_length - len(source_ids)
        candidate_ids = self.tokenizer.encode(
            self.cand_prefix + candidate,
            max_length=candidate_max_length,
            truncation=True,
        )
        return source_ids + candidate_ids

    def postprocess(self, outputs, input_ids: torch.Tensor):
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand_idxs = torch.where(input_ids == self.cand_prefix_id)
        cand_encs = encs[cand_idxs[0], cand_idxs[1], :]

        # reduce
        source_cand_encs = torch.cat([source_encs, cand_encs], dim=-1)
        return source_cand_encs.detach()

    def preprocess(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        #  <source_prefix_id>...<sep><cand_prefix_id>...<sep>
        if self.source_prefix_id is not None:
            assert all(
                [
                    self.source_prefix_id in input_ids[i]
                    for i in range(input_ids.shape[0])
                ]
            ), "<source> id not in input_ids"
        if self.cand_prefix_id is not None:
            assert all(
                [self.cand_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
            ), "<candidate> id not in input_ids"

        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        return input_ids, attention_mask

    @torch.no_grad
    def get_feature(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """Get the feature \phi(s, a) in a singleton form."""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        input_ids, attention_mask = self.preprocess(input_ids, attention_mask)

        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        return self.postprocess(outputs, input_ids)


class CustomBackbone(RMBackbone):
    @classmethod
    def from_pretrained(cls, model_name, device_map):
        inst = cls(model_name).to(device_map)
        return inst

    @property
    def device(self):
        return self.pretrained_model.device


class PythiaPretrained(CustomBackbone, nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = self.pretrained_model.config
        self.source_prefix_id = None
        self.cand_prefix_id = None

        self.eval()

    def tokenize_pair(
        self, prompt: str, candidate: str, source_max_length: int, max_length: int
    ):
        del source_max_length
        tokens = self.tokenizer.encode(
            prompt + candidate,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
        return tokens

    @torch.no_grad
    def get_feature(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        input_ids, attention_mask = self.preprocess(input_ids, attention_mask)

        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        return self.postprocess(outputs, attention_mask)

    def postprocess(self, outputs, attention_mask: torch.Tensor):
        encs = outputs.hidden_states[-1]
        last_pos = attention_mask.sum(-1).long() - 1
        batch_idx = torch.arange(len(encs), device=encs.device)
        return encs[batch_idx, last_pos, :].detach()


class SkyWorkRM(PythiaPretrained):
    def tokenize_pair(self, prompt, candidate, source_max_length=None, max_length=None):
        del source_max_length, max_length
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": candidate},
        ]
        conv_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
        if self.tokenizer.bos_token is not None and conv_formatted.startswith(
            self.tokenizer.bos_token
        ):
            conv_formatted = conv_formatted[len(self.tokenizer.bos_token) :]
        return self.tokenizer.encode(conv_formatted)


class DebertaV2Vanilla(CustomBackbone, nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = self.pretrained_model.config
        self.source_prefix_id = 1
        self.source_prefix = "[CLS]"
        self.cand_prefix_id = 2
        self.cand_prefix = "[SEP]"

        self.eval()

    def tokenize_pair(
        self, prompt: str, candidate: str, source_max_length: int, max_length: int
    ):
        source_ids = self.tokenizer.encode(
            self.source_prefix + prompt,
            max_length=source_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        candidate_max_length = max_length - len(source_ids)
        candidate_ids = self.tokenizer.encode(
            self.cand_prefix + candidate,
            max_length=candidate_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        return source_ids + candidate_ids


class DebertaV2PairRM(RMBackbone, DebertaV2PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.n_tasks = config.n_tasks
        self.drop_out = config.drop_out

        self.pretrained_model = DebertaV2Model(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)

        self.sep_token_id = config.sep_token_id  # to add
        self.source_prefix_id = config.source_prefix_id  # to add
        self.source_prefix = "<|source|>"  # to add
        self.cand_prefix_id = config.cand_prefix_id
        self.cand_prefix = "<|candidate|>"

        # Initialize weights and apply final processing
        self.post_init()
        self.eval()
