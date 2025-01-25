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

# Reference to https://github.com/OpenRLHF/OpenRLHF.

import logging
from typing import Optional

import deepspeed
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class LLM(nn.Module):
    """Large language model interface."""

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        device_map=None,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = (
                "flash_attention_2" if use_flash_attention_2 else "eager"
            )

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                logging.debug("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.LongTensor:
        generate_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }
        sequences = self.model.generate(**generate_args)
        return sequences

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids
        )

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs={"use_reentrant": False}
    ):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()


def build_critic_cls(base_cls, base_pretrain_cls, value_head_prefix):
    class CriticModel(base_pretrain_cls):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_cls(config))

            self.value_head_prefix = value_head_prefix
            setattr(
                self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False)
            )

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(
                -1
            )

            if return_output:
                return (values, outputs)
            else:
                return values

    return CriticModel


class Critic(nn.Module):
    """Large language model interface."""

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        init_value_head=True,
        value_head_prefix="score",
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            config = AutoConfig.from_pretrained(
                pretrain_or_model, trust_remote_code=True
            )
            config._attn_implementation = (
                "flash_attention_2" if use_flash_attention_2 else "eager"
            )
            value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)

            base_class = AutoModel._model_mapping[type(config)]
            critic_cls = build_critic_cls(
                base_class, base_class.__base__, value_head_prefix
            )

            self.model = critic_cls.from_pretrained(
                pretrain_or_model,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                quantization_config=nf4_config,
                device_map=device_map,
            )

            if init_value_head:
                value_head = getattr(self.model, value_head_prefix)
                if (
                    ds_config is not None
                    and ds_config["zero_optimization"]["stage"] == 3
                ):
                    with deepspeed.zero.GatheredParameters(
                        [value_head.weight], modifier_rank=0
                    ):
                        if torch.distributed.get_rank() == 0:
                            value_head.weight.data.normal_(
                                mean=0.0, std=1 / (config.hidden_size + 1)
                            )
                else:
                    value_head.weight.data.normal_(
                        mean=0.0, std=1 / (config.hidden_size + 1)
                    )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                logging.debug("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

    def forward(self, **input):
        return self.model(**input)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs={"use_reentrant": False}
    ):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
