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

"""SFT optimizer for imitation learning."""

from contextlib import nullcontext

import deepspeed
import torch
from torch.autograd.graph import save_on_cpu

from oat.learners.dap import DAPLearner
from oat.learners.offline_dap import OfflineDAPLearner
from oat.utils.lm_head import FusedLinear


class SFTLearner(DAPLearner):
    """Policy learning via supervised learning.

    We reuse the dap learner and take `chosen` as the target.
    """

    def learning_step(self, data):
        device = torch.cuda.current_device()
        chosen_ids, c_mask, _, _, extra = data
        chosen_ids = chosen_ids.squeeze(1).to(device)
        c_mask = c_mask.squeeze(1).to(device)
        context = save_on_cpu() if self.args.activation_offloading else nullcontext()
        with context:
            loss = self.model_forward(self.model, chosen_ids, c_mask, extra)
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
        infos = {
            "loss": loss.detach(),
            "chosen_reward": torch.zeros(1),
            "rejected_reward": torch.zeros(1),
        }
        return infos

    def model_forward(self, model, input_ids, att_masks, extra):
        prompt_id_lens = extra["prompt_ids_lens"]

        if len(prompt_id_lens) == 1:
            logits_to_keep = len(input_ids[0]) - prompt_id_lens[0]
            slice_indices = slice(-logits_to_keep, None)
            prompt_id_lens = [0 for _ in range(len(input_ids))]
        else:
            logits_to_keep = 0
            slice_indices = slice(None, None)

        model_output = model(
            input_ids,
            attention_mask=att_masks,
            logits_to_keep=logits_to_keep,
            without_logits=self.args.use_fused_lm_head,
        )
        if self.args.use_fused_lm_head:
            hidden_states = model_output.last_hidden_state[:, :-1]
            vocab_weights = model.model.lm_head.weight

            fused_linear = FusedLinear(compute_entropy=False)
            with deepspeed.zero.GatheredParameters(
                [vocab_weights],
                fwd_module=fused_linear,
                enabled=self.strategy.args.zero_stage == 3,
            ):
                print(
                    self.strategy.args.zero_stage == 3,
                    input_ids.shape,
                    att_masks.shape,
                    hidden_states.shape,
                    vocab_weights.shape,
                )
                target_logps, _ent = fused_linear(
                    hidden_states,
                    vocab_weights,
                    input_ids[:, 1:],
                    temperature=1,
                )
            del _ent
            completion_masks = att_masks.clone().bool()
            # mask prompts
            for mask, source_len in zip(completion_masks, prompt_id_lens):
                mask[:source_len] = False
            completion_masks = completion_masks[:, 1:]
            sum_loss = (target_logps * completion_masks).sum(-1)
            if not self.args.sft_sum_loss:
                sum_loss /= completion_masks.sum(-1)
            sft_loss = -sum_loss.mean()  # average across examples
        else:
            all_logits = model_output["logits"]
            batch_logps, _ = self.get_batch_logps(
                all_logits,
                input_ids[:, slice_indices],
                att_masks[:, slice_indices],
                prompt_id_lens,
                average_log_prob=not self.args.sft_sum_loss,
            )
            sft_loss = -batch_logps.mean()  # average across examples
        return sft_loss


class OfflineSFTLearner(SFTLearner, OfflineDAPLearner):
    """Offline learning."""
