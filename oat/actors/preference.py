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
import time
from typing import List

import numpy as np
import torch
import vllm

from oat.actors.base import ActorBase
from oat.exploration import ExplorationResults, Explorer, ModelBasedExplorer
from oat.rm import backbone, model
from oat.types import PreferenceData


class PreferenceActor(ActorBase):
    """The environment is a preference oracle. In this case the problem can be formulated
    as preference-based reinforcement learning (PbRL) or contextual dueling bandit (CDB).
    """

    def init(self, actor_id):
        super().init(actor_id)
        args = self.args
        assert (
            self.sampling_params.n >= 2
        ), "need to sample at least 2 responses per prompt"
        # ###################################
        # ####        Exploration        ####
        # ###################################
        self.learning_rm = False
        if args.exp_method == "no":
            if self.sampling_params.n > 2:
                logging.warn(
                    f"trying to sample {self.sampling_params.n} responses but "
                    "no selection mechanism is provided"
                )
        else:
            assert self.sampling_params.n > 2
            # We assume reward model-based explorer.
            rm_backbone_cls = backbone.get_cls(args.rm_backbone)
            logging.info(f"Using RM backbone {args.rm_backbone} {rm_backbone_cls}")
            self.rm_backbone = rm_backbone_cls.from_pretrained(
                args.rm_backbone, device_map="cuda:0"
            ).eval()

            explorer_cls = ModelBasedExplorer if args.model_rollout else Explorer
            self.explorer = explorer_cls(
                reward_model=getattr(model, args.exp_method)(args).cuda(),
                rm_backbone=self.rm_backbone,
                args=args,
            )

            if args.rm_pretrain:
                logging.info(f"Loading pretrained ENN from {args.rm_pretrain}")
                self.explorer.reward_model.load_state_dict(torch.load(args.rm_pretrain))
            else:
                self.learning_rm = True  # Learn RM online.
        self.model_rollout = args.model_rollout

        # ###################################
        # ####  Best-of-N for Evaluation ####
        # ###################################
        if args.best_of_n_eval:
            self.num_eval_gen = args.num_bon
        else:
            self.num_eval_gen = 1
        self.eval_sampling_params = vllm.SamplingParams(
            n=self.num_eval_gen,
            temperature=(
                args.eval_temperature
                if self.num_eval_gen == 1
                else args.bon_temperature
            ),
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
        )

    def generate(self, prompts: List[str], sampling_params: vllm.SamplingParams):
        outputs = super().generate(prompts, sampling_params)
        candidates = {}
        for i in range(len(outputs)):
            # for each prompt
            candidates[i] = []
            for k in range(sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text.strip())
        return candidates

    def generate_and_maybe_eval(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ):
        assert self.eval_mode
        candidates = self.generate(formatted_prompts, self.eval_sampling_params)

        if self.num_eval_gen > 1:
            # best of n sampling
            responses = self.explorer.best_of_n(prompts, candidates)
        else:
            responses = [candidates[i][0] for i in range(len(prompts))]

        if references:
            logging.debug(f"Evaluating using oracle {self.oracle}")
            st = time.time()
            win_probs, _ = self.oracle.compare(
                prompts,
                responses,
                references,
                batch_size=self.oracle_batch_size,
                return_probs=True,
                disable_tqdm=True,
            )
            logging.debug(f"Time elapse {time.time() - st}")
            return responses, win_probs
        return responses, None

    def online_eval(self, prompts, references, candidates):
        """Evaluate online responses."""
        win_probs_1, _ = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            references,
            batch_size=self.oracle_batch_size,
            return_probs=True,
            disable_tqdm=True,
        )
        win_probs_2, _ = self.oracle.compare(
            prompts,
            [candidates[i][1] for i in range(len(prompts))],
            references,
            batch_size=self.oracle_batch_size,
            return_probs=True,
            disable_tqdm=True,
        )
        return (win_probs_1 + win_probs_2) / 2

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[PreferenceData]:
        assert not self.eval_mode
        info = {}

        # step 1. generate
        st = time.time()
        all_candidates = self.generate(formatted_prompts, self.sampling_params)
        info["actor/generate_time"] = time.time() - st

        # step 2a. optional selection
        results = None
        if self.sampling_params.n > 2:
            results: ExplorationResults
            results = self.explorer.select(prompts, all_candidates)
            candidates = results.dueling_candidates
        else:
            candidates = all_candidates

        # step 2b. optional online eval
        if self.enable_online_evaluation:
            assert references is not None
            win_probs = self.online_eval(prompts, references, candidates)
            info["eval/online_win_probs"] = win_probs.mean()

        # step 3. query for oracle preference
        st = time.time()
        bt_probs, _ = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            [candidates[i][1] for i in range(len(prompts))],
            batch_size=self.oracle_batch_size,
            return_probs=True,
            disable_tqdm=True,
        )
        info["actor/first_action_win_prob"] = bt_probs.mean().item()
        info["actor/oracle_time"] = time.time() - st

        if self.args.bt_sample:
            binary_feedback = torch.bernoulli(torch.from_numpy(bt_probs)).bool().numpy()
        else:
            binary_feedback = bt_probs > 0.5

        chosen = 1 - binary_feedback

        # Model-based rollout for 1) Dyna - sample efficiency; 2) Better argmax r approximation.
        # (Mixed preference learning: Section 4.2.3 of https://arxiv.org/pdf/2411.01493)
        if self.model_rollout:
            # Record metric and overwrite label.
            model_data = np.array(results.is_model_data)
            model_rollout_correct = chosen[model_data] == 0
            model_rollout_acc = np.sum(model_rollout_correct) / (
                np.sum(model_data) + 1e-8
            )
            model_rollout_win_prob = np.nan_to_num(bt_probs[model_data].mean())
            info["eval/model_rollout_acc"] = model_rollout_acc
            info["eval/model_rollout_win_prob"] = model_rollout_win_prob

        rejected = 1 - chosen

        same_response = [
            candidates[i][chosen[i]] == candidates[i][rejected[i]]
            for i in range(len(prompts))
        ]

        if self.learning_rm:
            # Measure the internal RM accuracy
            pred_first_win, _ = self.explorer.compare(results.candidate_features)
            candidate_features = results.candidate_features.cpu()
            correct = pred_first_win == binary_feedback
            info["eval/rm_acc"] = correct.mean().item()

        if results is not None:
            info.update(results.info)

        chosen_responses = [candidates[i][chosen[i]] for i in range(len(prompts))]
        rejected_responses = [candidates[i][rejected[i]] for i in range(len(prompts))]

        preference_data = [
            PreferenceData(
                prompt=prompts[i],
                chosen_id=chosen[i],
                chosen_response=chosen_responses[i],
                rejected_response=rejected_responses[i],
                chosen_feature=(
                    candidate_features[i][chosen[i]] if self.learning_rm else None
                ),
                rejected_feature=(
                    candidate_features[i][rejected[i]] if self.learning_rm else None
                ),
                init_clash=results.init_clash[i] if self.learning_rm else False,
                loss_mask=not same_response[i],
                is_model_data=results.is_model_data[i] if self.learning_rm else False,
                info=info,
            )
            for i in range(len(prompts))
        ]

        handle = self.ipc_client.serialize_ipc(preference_data)
        return handle
