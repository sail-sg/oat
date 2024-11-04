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
import tree
import vllm

from oat import oracles
from oat.args import OATArgs
from oat.exploration import ExplorationResults, Explorer, ModelBasedExplorer
from oat.rm import backbone, model
from oat.types import PreferenceData
from oat.utils.distributed import WorkerWrap, torch_type_codec
from oat.utils.ipc import PlasmaShmClient

logging.getLogger("vllm").setLevel(logging.ERROR)


class Actor:
    """Actor handles the interaction between the LLM policy and the environment."""

    def __init__(self, ipc_server, vllm_args, args: OATArgs) -> None:
        self.args = args
        self.eval_mode = False
        self.pi_beta_weights = None
        # Measuring the **online** performance
        self.enable_online_evaluation = args.online_evaluation

        self.ipc_client = PlasmaShmClient(ipc_server)

        # ###################################
        # ####      vLLM Generation      ####
        # ###################################
        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            n=args.num_samples,
        )

        self.__vllm_version__ = vllm.__version__

        assert self.__vllm_version__ >= "0.4.1", "Upgrade to vLLM >= 0.4.1"
        assert (
            self.sampling_params.n >= 2
        ), "need to sample at least 2 responses per prompt"

        vllm.worker.worker.Worker = WorkerWrap
        vllm_args.update({"seed": time.time_ns() % 2**32})
        self.llm = vllm.LLM(**vllm_args)
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        # ###################################
        # ####    Oracle Reward Model    ####
        # ###################################
        oracle_cls = oracles.get_cls(args.reward_oracle)
        logging.info(f"Using reward oracle {args.reward_oracle} {oracle_cls}")
        self.oracle = oracle_cls(
            reward_model_path=args.reward_oracle,
            tokenizer_path=args.pretrain,
            remote_rm_url=args.remote_rm_url,  # Only for remote RM.
            max_workers=args.remote_rm_client_workers,  # Only for remote RM.
        )
        self.reward_oracle_batch_size = args.reward_oracle_batch_size

        # ###################################
        # ####        Exploration        ####
        # ###################################
        self.learning_rm = False
        if args.exp_method == "no":
            if self.sampling_params.n == 2:
                logging.warn(
                    f"trying to sample {self.sampling_params.n} responses but"
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
        outputs = self.llm.generate(
            prompts, sampling_params=sampling_params, use_tqdm=False
        )
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
        """
        1) Generate responses for given prompts;
        2) Optionally evaluate the win rate over references based on the oracle reward model.
        """
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
            win_probs = self.oracle.compare(
                prompts,
                responses,
                references,
                batch_size=self.reward_oracle_batch_size,
                return_probs=True,
                disable_tqdm=True,
            )
            logging.debug(f"Time elapse {time.time() - st}")
            return responses, win_probs
        return responses, None

    def online_eval(self, prompts, references, candidates):
        win_probs_1 = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            references,
            batch_size=self.reward_oracle_batch_size,
            return_probs=True,
            disable_tqdm=True,
        )
        win_probs_2 = self.oracle.compare(
            prompts,
            [candidates[i][1] for i in range(len(prompts))],
            references,
            batch_size=self.reward_oracle_batch_size,
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
        """Step the actor.

        Given a prompt x, K responses {y_1, ..., y_K} are sample from the behavior LLM pi_beta,
        from which 2 responses are selected to query the oracle for preference signal.
        The final constructed pair (x, y_w, y_l) is inserted into the replay buffer for learners.

        OATArgs:
            prompts: A list of prompt texts.
            formatted_prompts: A list of chat template formatted prompt texts.
            references: A list of reference texts.
        """
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
        bt_probs = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            [candidates[i][1] for i in range(len(prompts))],
            batch_size=self.reward_oracle_batch_size,
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

        # Model-based rollout for sampling efficiency.
        # (Mixed preference learning)
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
            pred_first_win = self.explorer.compare(results.candidate_features)
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
                same=same_response[i],
                is_model_data=results.is_model_data[i] if self.learning_rm else False,
                info=info,
            )
            for i in range(len(prompts))
        ]

        handle = self.ipc_client.serialize_ipc(preference_data)
        return handle

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        self._model_update_group = (
            self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self._stop_remote_worker_execution_loop()
        return self.llm.llm_engine.model_executor.driver_worker.update_weight(
            name, dtype, shape, empty_cache
        )

    def update_rm(self, name, dtype, shape):
        assert self.learning_rm
        dtype = torch_type_codec(dtype)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)
        params_dict = dict(self.explorer.reward_model.named_parameters())
        model.default_weight_loader(params_dict[name], weight)
        del weight

    def notify_eval_start(self):
        """Temporarily cache the current behavior policy weights to CPU."""
        self.eval_mode = True
        logging.debug("Start offloading...")
        st = time.time()
        self.cache_model_state = tree.map_structure(
            lambda x: x.cpu(), self.model.state_dict()
        )
        logging.debug(f"Finished offloading in {time.time() - st} seconds")

    def notify_eval_done(self):
        assert self.eval_mode
        logging.debug("Start loading from cpu...")
        st = time.time()
        self.model.load_state_dict(self.cache_model_state)
        logging.debug(f"Finished loading in {time.time() - st} seconds")
        self.eval_mode = False

    def _stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__vllm_version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()
