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

"""XPO: https://arxiv.org/pdf/2405.21046."""

from dataclasses import dataclass
from typing import List

import torch
import vllm

from oat.actors import PreferenceActor
from oat.args import OATArgs
from oat.learners.dap import DAPLearner
from oat.types import DAPAlgo


@dataclass
class XPOArgs(OATArgs):
    """Exploratory preference optimization arguments."""

    xpo_alpha: float = 5e-6
    xpo_offload_actor_ref: bool = False


class XPOActor(PreferenceActor):
    """Sample one response from llm and another from ref_llm."""

    def init(self, actor_id):
        super().init(actor_id)
        args = self.args
        self.sampling_params.n = 1  # one for each llm
        self.offload_ref_model = args.xpo_offload_actor_ref

        if not self.offload_ref_model:
            self.ref_llm = vllm.LLM(**self.vllm_args)
        else:
            self.ref_llm = None
            self.cache_ref_model_state = {
                k: v.cpu() for k, v in self.model.named_parameters()
            }

    def generate(self, prompts: List[str], sampling_params: vllm.SamplingParams):
        if self.eval_mode:
            return super().generate(prompts, sampling_params)

        assert sampling_params.n == 1
        candidates = {}

        for llm in [self.llm, self.ref_llm]:
            if llm is not None:
                outputs = llm.generate(
                    prompts, sampling_params=sampling_params, use_tqdm=False
                )
            else:
                # Cache current llm's weights, load ref_llm for infer and restore
                # original llm's weights.
                self.notify_eval_start(eval=False)
                self.model.load_state_dict(self.cache_ref_model_state)
                outputs = self.llm.generate(
                    prompts, sampling_params=sampling_params, use_tqdm=False
                )
                self.notify_eval_done(eval=False)
            for i in range(len(outputs)):
                # for each prompt
                if i not in candidates:
                    candidates[i] = []
                candidates[i].append(outputs[i].outputs[0].text.strip())

        return candidates


class XPOLearner(DAPLearner):
    """Additional optimism loss term: log(\pi(y_ref|x))."""

    def _init(self, args: XPOArgs, actors) -> None:
        super()._init(args, actors)
        assert self.algo == DAPAlgo.DPO and self.ref_model is not None
        self.xpo_alpha = args.xpo_alpha

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

        # `chosen` indicates the original sampling source:
        # 0 - rejected_ids are from the ref policy
        # 1 - chosen_ids are from the ref policy
        chosen = torch.tensor(extra["chosen_ids"]).to(device)
        ref_logps = torch.where(chosen == 0, rejected_logps, chosen_logps)
        optimism_loss = (ref_logps * loss_masks).mean()

        loss = preference_loss + self.xpo_alpha * optimism_loss
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

        infos = {
            "loss": loss.detach(),
            "optimism_loss": optimism_loss.detach(),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
        }
        return infos
