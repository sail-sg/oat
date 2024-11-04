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

import torch

from oat.learners.base import LearnerBase
from oat.learners.loss import DPOLoss, SimPOLoss
from oat.types import DAPAlgo
from oat.utils.data import pad_to_length


class DAPLearner(LearnerBase):
    """Direct Alignment from Preference (DAP) learning."""

    def _init(self, args, actors) -> None:
        super()._init(args, actors)

        if self.algo in [DAPAlgo.DPO, DAPAlgo.IPO, DAPAlgo.SLiC]:
            self.loss = DPOLoss(args.beta, args.label_smoothing, dap_algo=self.algo)
        elif self.algo == DAPAlgo.SimPO:
            self.loss = SimPOLoss(
                args.beta, args.gamma_beta_ratio, args.label_smoothing
            )
        else:
            raise ValueError("Invalid DAP Algorithm")

    def learning_step(self, data):
        device = torch.cuda.current_device()
        chosen_ids, c_mask, rejected_ids, r_mask, extra = data
        chosen_ids = chosen_ids.squeeze(1).to(device)
        c_mask = c_mask.squeeze(1).to(device)
        rejected_ids = rejected_ids.squeeze(1).to(device)
        r_mask = r_mask.squeeze(1).to(device)

        prompt_id_lens = extra["prompt_ids_lens"]
        loss_masks = 1 - torch.tensor(extra["same_masks"]).float().to(device)

        chosen_logps, rejected_logps, _ = self.concatenated_forward(
            self.model, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
        )
        if self.ref_model is not None:
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, _ = (
                    self.concatenated_forward(
                        self.ref_model,
                        chosen_ids,
                        c_mask,
                        rejected_ids,
                        r_mask,
                        prompt_id_lens,
                    )
                )
            preference_loss, chosen_reward, rejected_reward = self.loss(
                chosen_logps,
                rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                loss_masks,
            )
        else:
            preference_loss, chosen_reward, rejected_reward = self.loss(
                chosen_logps, rejected_logps, loss_masks
            )

        loss = preference_loss
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

        infos = {
            "loss": loss.detach(),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
        }
        return infos

    def concatenated_forward(
        self, model, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
    ):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks)
        all_logits = output["logits"]
        all_logps = self.get_batch_logps(
            all_logits,
            input_ids,
            att_masks,
            prompt_id_lens,
            average_log_prob=self.algo in [DAPAlgo.SimPO, DAPAlgo.IPO],
        )
        chosen_logps = all_logps[: chosen_ids.shape[0]]
        rejected_logps = all_logps[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss

    def concatenated_inputs(
        self, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
    ):
        """Concatenate the chosen and rejected inputs into a single tensor.

        OATArgs:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        max_length = max(chosen_ids.shape[1], rejected_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(rejected_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat(
            (
                pad_to_length(c_mask, max_length, 0),
                pad_to_length(r_mask, max_length, 0),
            ),
            dim=0,
        )
        return inputs_ids, att_masks, prompt_id_lens * 2

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        OATArgs:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        else:
            return (per_token_logps * loss_masks).sum(-1)
