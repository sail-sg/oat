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

import torch

from oat.learners.dap import DAPLearner
from oat.learners.offline_dap import OfflineDAPLearner


class SFTLearner(DAPLearner):
    """Policy learning via supervised learning.

    We reuse the dap learner and take `chosen` as the target.
    """

    def learning_step(self, data):
        device = torch.cuda.current_device()
        chosen_ids, c_mask, _, _, extra = data
        chosen_ids = chosen_ids.squeeze(1).to(device)
        c_mask = c_mask.squeeze(1).to(device)

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

        output = model(input_ids, attention_mask=att_masks)
        all_logits = output["logits"]
        all_logps, _ = self.get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=True
        )
        sft_loss = -all_logps.mean()  # average across examples
        return sft_loss


class OfflineSFTLearner(SFTLearner, OfflineDAPLearner):
    """Offline learning."""
