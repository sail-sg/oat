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

import pdb
import pickle

from transformers.trainer import get_scheduler

from oat.args import OATArgs, default_args_validation, get_default_args
from oat.model import LLM
from oat.utils.deepspeed import get_strategy


def main(args: OATArgs):
    assert args.lora_rank > 0
    model = LLM(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        # ds_config=strategy.get_ds_train_config(is_wrapped=True),
    )
    args, strategy = get_strategy(args)
    optimizer = strategy.create_optimizer(
        model,
        lr=args.learning_rate,
        betas=(args.adam_beta_1, args.adam_beta_2),
        weight_decay=args.l2,
    )
    scheduler = get_scheduler(
        "constant",
        optimizer,
    )
    (model, optimizer, scheduler) = strategy.prepare(
        (model, optimizer, scheduler),
    )
    unwrapped_model = strategy._unwrap_model(model)
    unwrapped_model.merge_adapter()
    state_dict = unwrapped_model.state_dict()
    # Remove base_model and base_layer prefixes
    state_dict = {
        k.removeprefix("base_model.model.").replace(".base_layer", ""): v
        for k, v in state_dict.items()
    }
    # Remove values with adapter prefix (example: "_lora")
    state_dict = {
        k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k
    }
    # When module to save, remove its prefix and discard the original module
    state_dict = {
        k.replace("modules_to_save.default.", ""): v
        for k, v in state_dict.items()
        if "original_module" not in k
    }
    pickle.dump(state_dict.keys(), open("lora_keys.pkl", "wb"))
    pdb.set_trace()


if __name__ == "__main__":
    args = get_default_args()
    args.gpus = 1
    args.lora_rank = 8
    args.pretrain = "Qwen/Qwen2.5-1.5B"
    args = default_args_validation(args)
    main(args)
