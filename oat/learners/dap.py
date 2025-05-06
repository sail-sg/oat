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

"""Direct optimizer - DAP: Direct Alignment from Preferences."""

import time
from typing import List, Tuple

import numpy as np
import torch
import tree
from torch import distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.learners.base import LearnerBase
from oat.learners.loss import BNFLoss, DPOLoss, SimPOLoss
from oat.model import LLM
from oat.types import DAPAlgo, PreferenceData, SFTAlgo
from oat.utils.data import PreferenceDataset, pad_to_length
from oat.utils.ops import disable_dropout


class DAPLearner(LearnerBase):
    """Direct Alignment from Preference (DAP) learning."""

    def _init(self, args: OATArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)

        if self.algo not in [DAPAlgo.SimPO, SFTAlgo.SFT]:
            self.strategy.print("Running reference-based algorithm... (DPO, IPO, etc.)")
            assert args.ref_pretrain, "Reference model must be non-empty"
            self.ref_model = LLM(
                args.ref_pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
            )
            disable_dropout(self.ref_model)
        else:
            self.strategy.print(
                f"Running reference-free algorithm... ({self.algo.name})"
            )

        # prepare models/optimizers...
        if self.algo not in [DAPAlgo.SimPO, SFTAlgo.SFT]:
            ((self.model, self.optimizer, self.scheduler), self.ref_model) = (
                self.strategy.prepare(
                    (self.model, self.optimizer, self.scheduler),
                    self.ref_model,
                )
            )
        else:
            (self.model, self.optimizer, self.scheduler) = self.strategy.prepare(
                (self.model, self.optimizer, self.scheduler),
            )
            self.ref_model = None

        if self.algo in [DAPAlgo.DPO, DAPAlgo.LR_DPO, DAPAlgo.IPO, DAPAlgo.SLiC]:
            self.loss = DPOLoss(
                beta=args.beta,
                label_smoothing=args.label_smoothing,
                dpo_positive_lambda=args.dpo_positive_lambda,
                len_reg_alpha=args.len_reg_alpha,
                sft_weight=args.sft_weight,
                dap_algo=self.algo,
            )
        elif self.algo == DAPAlgo.SimPO:
            self.loss = SimPOLoss(
                args.beta, args.gamma_beta_ratio, args.label_smoothing
            )
        elif self.algo == DAPAlgo.BNF:
            self.loss = BNFLoss()
        else:
            assert self.algo in SFTAlgo, "Invalid DAP Algorithm"

        self.dataset_builder = PreferenceDataset
        dist.barrier()

    def process_feedback_data(self, data_list: List[PreferenceData]):
        self.query_step += np.sum([not p.is_model_data for p in data_list])
        for pref in data_list:
            self.pi_buffer.append(pref)
            if self.args.dump_all_buffer:
                c = pref.chosen_response
                r = pref.rejected_response
                self.all_buffer.append(
                    PreferenceData(
                        prompt=pref.prompt,
                        chosen_response=c,
                        rejected_response=r,
                        same=c == r,
                    )
                )

    def learn(self, learning_round: int):
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.args.prompt_max_length,
            self.args.generate_max_length,
            self.strategy,
        )
        if learning_round == 1:
            self.strategy.print("Training example")
            self.strategy.print(dataset[0])

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.train_batch_size_per_device,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        for epoch in range(self.args.max_epochs):
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )
            acc_mean = []
            loss_mean = []
            chosen_rewards = []
            rejected_rewards = []
            reward_margin = []
            learn_batch_time = []

            self.model.train()
            st = time.time()
            for data in dataloader:
                if local_sgd_steps > self.args.max_sgd_steps:
                    break
                infos = self.learning_step(data)

                # metrics
                loss = infos.pop("loss")
                chosen_reward = infos.pop("chosen_reward")
                rejected_reward = infos.pop("rejected_reward")
                chosen_rewards.append(chosen_reward.mean().item())
                rejected_rewards.append(rejected_reward.mean().item())
                acc_mean.append((chosen_reward > rejected_reward).float().mean().item())
                loss_mean.append(loss.cpu().item())
                reward_margin.append((chosen_reward - rejected_reward).mean().item())

                step_bar.update()
                self.global_step += 1
                if self.global_step % self.strategy.accumulated_gradient == 0:
                    learn_batch_time.append(time.time() - st)
                    self.gradient_update_elapse = time.time() - self.gradient_update_st
                    st = time.time()
                    self.gradient_update_st = time.time()
                    self.policy_sgd_step += 1
                    local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "epoch": epoch + 1,
            "chosen_reward": np.mean(chosen_rewards),
            "rejected_reward": np.mean(rejected_rewards),
            "acc_mean": np.mean(acc_mean),
            "loss_mean": np.mean(loss_mean),
            "reward_margin": np.mean(reward_margin),
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        return train_info

    def learning_step(self, data):
        device = torch.cuda.current_device()
        chosen_ids, c_mask, rejected_ids, r_mask, extra = data
        chosen_ids = chosen_ids.squeeze(1).to(device)
        c_mask = c_mask.squeeze(1).to(device)
        rejected_ids = rejected_ids.squeeze(1).to(device)
        r_mask = r_mask.squeeze(1).to(device)

        prompt_id_lens = extra["prompt_ids_lens"]
        loss_masks = torch.tensor(extra["loss_masks"]).float().to(device)

        if self.algo == DAPAlgo.BNF:

            policy_logps, policy_entropy, token_masks = self.concatenated_forward(
                self.model, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
            )

            with torch.no_grad():
                ref_logps, _, _ = self.concatenated_forward(
                    self.ref_model,
                    chosen_ids,
                    c_mask,
                    rejected_ids,
                    r_mask,
                    prompt_id_lens,
                )
            # BNFLoss
            preference_loss, chosen_reward, rejected_reward = self.loss(
                policy_logps,
                policy_entropy,
                ref_logps,
                token_masks,
                loss_masks,
                chosen_ids.shape,
            )

        else:
            chosen_logps, rejected_logps, _, token_masks = self.concatenated_forward(
                self.model, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
            )

            if self.ref_model is not None:
                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps, _, _ = (
                        self.concatenated_forward(
                            self.ref_model,
                            chosen_ids,
                            c_mask,
                            rejected_ids,
                            r_mask,
                            prompt_id_lens,
                        )
                    )
                # DPOLoss
                preference_loss, chosen_reward, rejected_reward = self.loss(
                    chosen_logps,
                    rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    loss_masks,
                    token_masks,
                )
            else:
                # SimPOLoss
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

        if self.algo != DAPAlgo.BNF:

            all_logps, token_masks = self.get_batch_logps(
                all_logits,
                input_ids,
                att_masks,
                prompt_id_lens,
                average_log_prob=self.algo
                in [DAPAlgo.SimPO, DAPAlgo.IPO, DAPAlgo.LR_DPO],
            )
            chosen_logps = all_logps[: chosen_ids.shape[0]]
            rejected_logps = all_logps[chosen_ids.shape[0] :]
            aux_loss = output.aux_loss if "aux_loss" in output else []

            return (
                chosen_logps,
                rejected_logps,
                aux_loss,
                token_masks,
            )

        else:

            all_logps, entropy, token_masks = self.get_batch_logps(
                all_logits,
                input_ids,
                att_masks,
                prompt_id_lens,
            )

            return all_logps, entropy, token_masks

    def concatenated_inputs(
        self, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
    ):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
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

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
        prompt_id_lens: List[int],
        average_log_prob: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Get masked sum/avg log probabilities."""
        all_logp, target_logps, completion_masks = super().get_batch_logps(
            logits, labels, attention_mask, prompt_id_lens
        )
        if self.algo != DAPAlgo.BNF:
            length = completion_masks.sum(-1)
            if average_log_prob:
                return (target_logps * completion_masks).sum(
                    -1
                ) / length, completion_masks
            else:
                return (target_logps * completion_masks).sum(-1), completion_masks
        else:
            entropy = (all_logp.exp().detach() * all_logp).sum(
                -1
            ) - target_logps.exp().detach() * target_logps
            return (
                target_logps * completion_masks,
                entropy * completion_masks,
                completion_masks,
            )
