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

"""Next-token Prediction for continual pre-training."""

from typing import Tuple

import torch

from oat.learners.dap import DAPLearner
from oat.learners.offline_dap import OfflineDAPLearner


class NTPLearner(DAPLearner):
    """Continual pre-training via next-token prediction loss.

    We reuse the dap learner and take `chosen` as the target.
    """

    def learning_step(self, data):
        device = torch.cuda.current_device()
        chosen_ids, c_mask, _, _, _ = data
        chosen_ids = chosen_ids.squeeze(1).to(device)
        c_mask = c_mask.squeeze(1).to(device)

        loss = self.model_forward(self.model, chosen_ids, c_mask)
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

        infos = {
            "loss": loss.detach(),
            "chosen_reward": torch.zeros(1),
            "rejected_reward": torch.zeros(1),
        }
        return infos

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)

        Returns:
            all_logp: all log prob of shape (batch_size, sequence_length, vocab_size)
            target_logps: target log prob of shape (batch_size, sequence_length)
            completion_masks: mask=True if it is completion's token, shape (batch_size, sequence_length)
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        completion_masks = attention_mask.clone().bool()
        completion_masks = completion_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[completion_masks == False] = 0

        all_logp = logits.log_softmax(-1)
        target_logps = torch.gather(all_logp, dim=2, index=labels.unsqueeze(2)).squeeze(
            2
        )

        return all_logp, target_logps, completion_masks

    def model_forward(self, model, input_ids, att_masks):

        output = model(input_ids, attention_mask=att_masks)
        all_logits = output["logits"]
        _, target_logps, completion_masks = self.get_batch_logps(
            all_logits, input_ids, att_masks
        )
        target_logps_sum = (target_logps * completion_masks).sum(
            -1
        )  # NOTE: .sum instead of .mean
        sft_loss = -target_logps_sum.mean()  # average across examples
        return sft_loss


class OfflineNTPLearner(NTPLearner, OfflineDAPLearner):
    """Offline learning."""
