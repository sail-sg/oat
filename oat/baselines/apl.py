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

"""APL: https://arxiv.org/pdf/2402.08114.

Due to its design of using LLM as the reward model, we have to make the actor-
learner interface more complicated. We first generate responses and estimate
the entropy in actor, then compute the implicit reward margin in learner, and
finally get oracle feedback in actor.
"""

import time
from typing import Dict, List, Tuple

import launchpad as lp
import Levenshtein
import numpy as np
import torch
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from vllm.outputs import RequestOutput

from oat import actor
from oat.learners.dap import DAPLearner
from oat.model import LLM
from oat.types import Metric, PreferenceData
from oat.utils.data import zero_pad_sequences
from oat.utils.ipc import DataID, PlasmaShmClient


class APLActor(actor.Actor):
    """Sample a large batch and filter with entropy and reward margin."""

    def __init__(self, ipc_server, vllm_args, args: actor.OATArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.sampling_params.logprobs = 1

    def generate_and_entropy_filter(self, prompts: List[str]) -> DataID:
        assert not self.eval_mode
        # Generate.
        outputs = self.llm.generate(
            prompts, sampling_params=self.sampling_params, use_tqdm=False
        )

        ent_filtered_indices = None
        if not self.args.apl_pref_certainty_only:
            # Predictive entropy estimation.
            entropy_estimations = []
            for output in outputs:
                entropy = 0
                for resp_output in output.outputs:
                    entropy += resp_output.cumulative_logprob
                entropy /= len(output.outputs)
                entropy_estimations.append(entropy)
            ent_filtered_indices = np.argsort(entropy_estimations)[
                -self.args.pi_buffer_maxlen_per_device :
            ]  # Online and on-policy; as stated in their Appendix D.
            outputs = [outputs[i] for i in ent_filtered_indices]

        handle = self.ipc_client.serialize_ipc([outputs, ent_filtered_indices])
        return handle

    def query_oracle(self, handle: DataID):
        assert not self.eval_mode
        info = dict()
        prompts, candidates = self.ipc_client.deserialize_ipc(handle)
        bt_probs = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            [candidates[i][1] for i in range(len(prompts))],
            return_probs=True,
            disable_tqdm=True,
        )

        if self.args.bt_sample:
            binary_feedback = torch.bernoulli(torch.from_numpy(bt_probs)).bool().numpy()
        else:
            binary_feedback = bt_probs > 0.5
        chosen = 1 - binary_feedback
        rejected = 1 - chosen

        same_response = [
            candidates[i][chosen[i]] == candidates[i][rejected[i]]
            for i in range(len(prompts))
        ]

        preference_data = [
            PreferenceData(
                prompt=prompts[i],
                chosen_id=chosen[i],
                chosen_response=candidates[i][chosen[i]],
                rejected_response=candidates[i][rejected[i]],
                chosen_feature=None,
                rejected_feature=None,
                init_clash=False,
                same=same_response[i],
                is_model_data=False,
                info=info,
            )
            for i in range(len(prompts))
        ]

        metric = {
            "actor/chosen_avg_str_len": np.mean(
                [len(p.chosen_response) for p in preference_data]
            ),
            "actor/rejected_avg_str_len": np.mean(
                [len(p.rejected_response) for p in preference_data]
            ),
            "actor/init_clash_ratio": np.mean([p.init_clash for p in preference_data]),
            "actor/same_response_ratio": np.mean([p.same for p in preference_data]),
            "actor/pair_edit_dist": np.mean(
                [
                    Levenshtein.distance(p.chosen_response, p.rejected_response)
                    for p in preference_data
                ]
            ),
            "actor/model_data_ratio": np.mean(
                [p.is_model_data for p in preference_data]
            ),
            "actor/chosen_id": np.mean([p.chosen_id for p in preference_data]),
            "actor/first_action_win_prob": bt_probs.mean().item(),
        }

        handle = self.ipc_client.serialize_ipc([preference_data, metric])
        return handle


class APLLearner(DAPLearner):
    def run(self):
        """Overriding the learner run loop for APL."""
        self.ipc_client = PlasmaShmClient(self.ipc_server)
        self._init(self.args, self.actors)

        self.steps = 0
        self.start_time = time.time()

        self.actor_info = {}

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True)

        self.steps = 1
        self.gradient_update_st = time.time()
        for p_ep in range(self.args.num_prompt_epoch):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(p_ep)
                self.strategy.print(f"Set DistributedSampler at epoch {p_ep}")
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
            )

            for processed_prompts, raw_prompts, refs in self.prompts_dataloader:
                # ###################### #
                # (BEGIN) Logic for APL  #
                # ###################### #

                # APL Algo 1, Line 7-8: generate response & (optionally) filter by entropy.
                st_time = time.time()
                rank = torch.distributed.get_rank()
                actor: APLActor = self.actors[rank % len(self.actors)]
                handle = actor.generate_and_entropy_filter(processed_prompts)
                outputs: List[RequestOutput]
                outputs, ent_filtered_indices = self.ipc_client.deserialize_ipc(handle)

                # APL Algo 1, Line 8-9: get implicit reward margin and select pairs.
                output_info1 = f"({len(outputs)},{len(outputs[0].outputs)})"
                if not self.args.apl_pref_certainty_only:
                    # Keep all filtered prompts; select response pair.
                    processed_prompts = [
                        processed_prompts[i] for i in ent_filtered_indices
                    ]
                    raw_prompts = [raw_prompts[i] for i in ent_filtered_indices]
                    candidates, info = implicit_reward_filtering_response_only(
                        self.model,
                        self.ref_model,
                        self.tokenizer,
                        outputs,
                    )
                else:
                    # Select the (x, y, y') triplet.
                    processed_prompts, raw_prompts, candidates, info = (
                        implicit_reward_filtering_triplet(
                            processed_prompts,
                            raw_prompts,
                            self.model,
                            self.ref_model,
                            self.tokenizer,
                            outputs,
                            self.args.pi_buffer_maxlen_per_device,
                        )
                    )
                output_info2 = f"({len(processed_prompts)},{len(candidates[0])})"
                # APL Algo 1, Line 10: query oracle RM.
                handle = actor.query_oracle(
                    self.ipc_client.serialize_ipc([processed_prompts, candidates])
                )
                preference_data: List[PreferenceData]
                preference_data, self.actor_info = self.ipc_client.deserialize_ipc(
                    handle
                )
                self.actor_info.update(
                    {
                        "actor/generate_time": time.time() - st_time,
                        **info,
                    }
                )

                # ###################### #
                #   (END) Logic for APL  #
                # ###################### #

                self.prompt_consumed += len(refs)
                self.query_step += np.sum(
                    [not p.is_model_data for p in preference_data]
                )
                self.process_preference_data(preference_data, raw_prompts)

                if self.steps % self.update_interval == 0:
                    train_info = self.preference_learning(
                        self.steps // self.update_interval
                    )

                    self.eval_and_log(train_info)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        self.sync_params_to_actors()

                    if (
                        self.steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                progress_bar.update()
                self.steps += 1

        self.eval_and_log(train_info, eval=True)

        if self.strategy.is_rank_0():
            self._wandb.finish()
            lp.stop()


def implicit_reward_filtering_response_only(
    policy_model: LLM,
    ref_model: LLM,
    tokenizer: PreTrainedTokenizer,
    outputs: List[RequestOutput],
) -> Tuple[List[str], Dict[str, List[str]], Metric]:
    """Select the response pair that gives the largest implicit reward margin."""
    candidates = {}

    avg_margins = []
    selected_margins = []
    for i, output in enumerate(outputs):
        # for each prompt
        prompt_response_ids = [
            torch.tensor(output.prompt_token_ids + list(o.token_ids))
            for o in output.outputs
        ]
        prompt_response_masks = [torch.ones_like(ids) for ids in prompt_response_ids]

        prompt_response_ids = zero_pad_sequences(
            prompt_response_ids, side="right", value=tokenizer.pad_token_id
        )
        prompt_response_masks = zero_pad_sequences(prompt_response_masks, side="right")

        prompt_response_ids = prompt_response_ids.cuda()
        prompt_response_masks = prompt_response_masks.cuda()

        logprobs = compute_logp(
            policy_model,
            prompt_response_ids,
            prompt_response_masks,
            len(output.prompt_token_ids),
        )

        logprobs_ref = compute_logp(
            ref_model,
            prompt_response_ids,
            prompt_response_masks,
            len(output.prompt_token_ids),
        )
        M = len(prompt_response_ids)
        implicit_rewards = logprobs - logprobs_ref
        # NOTE the above will be zero until the policy is updated, need to avoid
        # selecting the same response during argmax, so subtract identity.
        reward_margins = torch.abs(
            implicit_rewards.view(M, 1) - implicit_rewards.view(1, M)
        ) - torch.eye(M, device=implicit_rewards.device)

        max_idx = reward_margins.argmax()
        pair_indices = [max_idx // M, max_idx % M]
        candidates[i] = [output.outputs[j].text for j in pair_indices]

        avg_margins.append(reward_margins.mean().cpu().item())
        selected_margins.append(reward_margins.max().cpu().item())

    return (
        candidates,
        {
            "actor/avg_margins": np.mean(avg_margins),
            "actor/selected_margins": np.mean(selected_margins),
        },
    )


def implicit_reward_filtering_triplet(
    processed_prompts: List[str],
    raw_prompts: List[str],
    policy_model: LLM,
    ref_model: LLM,
    tokenizer: PreTrainedTokenizer,
    outputs: List[RequestOutput],
    num_keep: int,
) -> Tuple[List[str], Dict[str, List[str]], Metric]:
    """Select the response pair that gives the largest implicit reward margin."""
    scores = []

    for output in outputs:
        # for each prompt
        prompt_response_ids = [
            torch.tensor(output.prompt_token_ids + o.token_ids) for o in output.outputs
        ]
        assert len(prompt_response_ids) == 2, len(prompt_response_ids)
        prompt_response_masks = [torch.ones_like(ids) for ids in prompt_response_ids]

        prompt_response_ids = zero_pad_sequences(
            prompt_response_ids, side="right", value=tokenizer.pad_token_id
        )
        prompt_response_masks = zero_pad_sequences(prompt_response_masks, side="right")

        prompt_response_ids = prompt_response_ids.cuda()
        prompt_response_masks = prompt_response_masks.cuda()

        logprobs = compute_logp(
            policy_model,
            prompt_response_ids,
            prompt_response_masks,
            len(output.prompt_token_ids),
        )

        logprobs_ref = compute_logp(
            ref_model,
            prompt_response_ids,
            prompt_response_masks,
            len(output.prompt_token_ids),
        )
        implicit_rewards = logprobs - logprobs_ref
        scores.append(torch.abs(implicit_rewards[0] - implicit_rewards[1]).cpu().item())

    scores = np.array(scores)
    top_indices = np.argsort(scores)[-num_keep:].tolist()

    processed_prompts = [processed_prompts[idx] for idx in top_indices]
    raw_prompts = [raw_prompts[idx] for idx in top_indices]
    candidates = {
        i: [outputs[idx].outputs[0].text.strip(), outputs[idx].outputs[1].text.strip()]
        for i, idx in enumerate(top_indices)
    }
    info = {
        "actor/avg_scores": scores.mean(),
        "actor/selected_scores": scores[top_indices].mean(),
    }

    return (
        processed_prompts,
        raw_prompts,
        candidates,
        info,
    )


@torch.no_grad
def compute_logp(model, prompt_response_ids, prompt_response_masks, prompt_len: int):
    model_output = model(prompt_response_ids, attention_mask=prompt_response_masks)
    all_logits = model_output["logits"]
    prompt_id_lens = [prompt_len] * len(prompt_response_masks)
    return get_batch_logps(
        all_logits,
        prompt_response_ids,
        prompt_response_masks,
        prompt_id_lens,
        average_log_prob=False,
    )


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask,
    prompt_id_lens,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
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
