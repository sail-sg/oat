
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import functools
import itertools
import logging
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, TimeoutError
from typing import Any, List, Literal, Tuple, Optional, Dict

import numpy as np
import torch
import tree

from datasets import concatenate_datasets, load_from_disk,load_dataset
from torch.utils.data import DataLoader
from collections import defaultdict
from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.oracles.cal_oracle import CALOracle
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric, TrajectoryData
from oat.utils.data import PromptDataset, TrajectoryDataset, load_data_from_disk_or_hf
from oat.utils.math_grader import (
    answer_tag_reward_fn,
    boxed_reward_fn,
    r1_distill_qwen_math_reward_fn,
)
from oat.utils.ops import masked_mean, masked_sum

"""
1. To do RL from base models, we use proper prompt template to make the base model answer questions.
"""


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


def apply_no_template(question: str):
    return question


def apply_r1_distill_qwen_template(question: str):
    # https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations
    question = (
        question
        + " Please reason step by step, and put your final answer within \\boxed{}.\n"
    )
    return "<｜begin▁of▁sentence｜><｜User｜>" + question + "<｜Assistant｜><think>\n"


TEMPLATE_FACTORY = {
    "qwen_math": apply_qwen_math_template,
    "r1": apply_r1_template,
    "no": apply_no_template,
    "r1_distill_qwen": apply_r1_distill_qwen_template,
}


"""
2. To train reasoning models that solve math questions, we need to define an oracle (environment) that provides rule-based verification rewards.
We instantiate the oracle based on Oat's OracleBase and implement the grading logic.
"""


class MATHOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the math answer grading."""

    def __init__(
        self, template, verifier_version, correct_reward, incorrect_reward
    ) -> None:
        super().__init__()
        if template == "r1":
            math_reward_fn = answer_tag_reward_fn
        elif template == "r1_distill_qwen":
            math_reward_fn = r1_distill_qwen_math_reward_fn
        else:
            math_reward_fn = boxed_reward_fn

        self.math_reward_fn = functools.partial(
            math_reward_fn,
            fast=verifier_version == "fast",
            correct_reward=correct_reward,
            incorrect_reward=incorrect_reward,
        )
        self.incorrect_reward = incorrect_reward

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, List[Any]]:
        del inputs, batch_size

        rewards = []
        infos = []
        for resp, ref in zip(responses, references):
            info, r = self.math_reward_fn(resp, ref)
            rewards.append(r)
            infos.append(info)

        return torch.tensor(rewards), infos

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[np.ndarray, List[Any]]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info


"""
2. Define extra arguments needed besides Oat's PPOArgs, mainly about choosing the prompt template.
"""


@dataclass
class CALArgs(PPOArgs):
    """
    Our custom arguments, inheriting from PPOArgs to get all the standard
    PPO settings, plus our new CAL-specific ones.
    """
    use_cal_oracle: bool = field(
        default=False,
        metadata={"help": "If True, use the custom Credit Assignment LLM Oracle."}
    )
    cal_model_name: str = field(
        default="gemini-1.5-flash-latest",
        metadata={"help": "The model name for the CAL."}
    )
    cal_few_shot_path: str = field(
        default="cal_examples.json",
        metadata={"help": "Path to the JSON file with few-shot examples for the CAL."}
    )
    negative_reward: float = field(
        default=-1.0,
        metadata={"help": "The negative reward value assigned to the error segment."}
    )
    prompt_template: Literal["qwen_math", "r1", "r1_distill_qwen", "no"] = field(
        default="no",
        metadata={"help": "How to wrap each question before sending it to the model."}
    )
"""
3. Instantiate the actor based on Oat's PPOActor, which controls the reasoning trace generation (`self.sampling_params`) and the rewarding (`self.oracle`).
"""


class CalActor(PPOActor): 
    def init(self, actor_id, save_path):
        logging.info(f"--- [CalActor {actor_id}, PID: {os.getpid()}] Starting init ---")
        super().init(actor_id, save_path)

        # This is the key change: we conditionally instantiate the oracle
        if self.args.use_cal_oracle:
            logging.info(f"--- [CalActor {actor_id}, PID: {os.getpid()}] Initializing CALOracle... ---")
            self.oracle = CALOracle(
                cal_model_name=self.args.cal_model_name,
                few_shot_path=self.args.cal_few_shot_path,
            )
        else:
            # Fallback to the standard MATHOracle for baseline runs
            logging.info(f"--- [CalActor {actor_id}, PID: {os.getpid()}] Initializing standard MATHOracle... ---")
            from oat.oracles.math import MATHOracle
            self.oracle = MATHOracle(
                template=self.args.prompt_template,
                verifier_version=self.args.verifier_version,
                correct_reward=self.args.correct_reward,
                incorrect_reward=self.args.incorrect_reward,
            )
         # After the oracle is ready, signal that this actor is ready to proceed.
        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()
        # logging.info(f"--- [CalActor {actor_id}, PID: {os.getpid()}] Oracle initialized. ---")
        
        # After the oracle is ready, signal that this actor is ready to proceed.
        # Do not add a barrier here as it will cause a deadlock.
        # The synchronization should be handled by the learner process.

        if self.args.prompt_template in ["qwen_math", "no"]:
            self.sampling_params.stop = None
            self.sampling_params.stop_token_ids = None
            self.eval_sampling_params.stop = None
            self.eval_sampling_params.stop_token_ids = None
        elif self.args.prompt_template == "r1":
            # Let's stop when the model completes its answer.
            self.sampling_params.stop = ["</answer>"]
            self.sampling_params.include_stop_str_in_output = True
            self.eval_sampling_params.stop = ["</answer>"]
            self.eval_sampling_params.include_stop_str_in_output = True

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: Optional[List[str]] = None,
    ) -> List[TrajectoryData]:
        """Main logic for the actor to generate trajectories (reasoning traces)."""
        assert not self.eval_mode
        logging.info(f"--- [CalActor, PID: {os.getpid()}] Entered step method. ---")
        info = {}

        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)

        candidates = []
        prompt_token_ids = []
        no_eos = []
        response_ids = []
        response_logprobs = []
        resp_lens = []
        for i in range(len(outputs)):
            # for each prompt
            prompt_token_ids.append(outputs[i].prompt_token_ids)
            candidates.append([])
            response_logprobs.append([])
            response_ids.append([])
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)
                no_eos.append(outputs[i].outputs[k].finish_reason == "length")
                token_ids = outputs[i].outputs[k].token_ids
                logps = outputs[i].outputs[k].logprobs
                logps = [item[token_ids[i]].logprob for i, item in enumerate(logps)]
                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                resp_lens.append(len(token_ids))

        info["actor/generate_time"] = time.time() - st

        # step 2. verify
        st = time.time()
        rewards, oracle_infos = self.oracle.get_reward(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in prompts
                )
            ),
            tree.flatten(candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in references
                )
            ),
        )

        info["actor/verify_time"] = time.time() - st
        logging.info(f"actor dummy reward {rewards.mean()}")
        info["actor/rewards"] = rewards.mean().item()
        info["actor/num_data"] = rewards.numel()
        info["actor/formatted"] = np.mean([i["formatted"] for i in oracle_infos])
        info["actor/response_tok_len"] = np.mean(resp_lens)
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = self.sampling_params.temperature

        rewards = rewards.reshape(len(prompts), -1)
        no_eos = np.array(no_eos).reshape(len(prompts), -1)
        info["actor/no_eos_count"] = no_eos.sum()

        # Reshape oracle_infos to match the (prompts, num_samples) structure
        oracle_infos = np.array(oracle_infos).reshape(len(prompts), -1)
        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                reward = rewards[i][j].item()
                if no_eos[i][j]:
                    # Set zero reward for truncated outputs.
                    reward = 0
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                trajectory_data.append(
                    TrajectoryData(
                        prompt=prompt,
                        prompt_ids=prompt_token_ids[i],
                        response=candidates_per_prompt[j],
                        response_ids=response_ids[i][j],
                        response_logprobs=response_logprobs[i][j],
                        rewards=dense_rewards,
                        loss_mask=not no_eos[i][j] if self.args.ignore_no_eos else True,
                        info=oracle_infos[i][j],
                    )
                )
        logging.info(f"actor finished data_len={len(trajectory_data)}")
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle


"""
4. Instantiate the learner based on PPOLearner. Here we adapt the `evaluate` logic to run multiple math benchmarks.
"""

class CalLearner(PPOLearner):
    """
    A custom PPOLearner that uses a Credit Assignment LLM (CAL) for
    fine-grained credit assignment, replacing the standard PPO critic network.
    """

    def _init(self, args: CALArgs, actors: List[ActorBase]) -> None:
        """
        Initializes the CalLearner. We override this to PREVENT the
        initialization of the critic network, saving memory and compute.
        """

        super(PPOLearner, self)._init(args, actors)
        self.args = args
        self.dataset_builder = TrajectoryDataset

        self.critic = None
        logging.info("CalLearner initialized WITHOUT a critic network.")
        
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )

    def _map_segment_to_token_indices(self, response_ids: List[int], error_segment: str) -> tuple[int, int]:
        """
        A helper function to map a string segment to token indices in a full token sequence.
        This is a non-trivial but crucial part of the implementation.
        """
        if not error_segment:
            return -1, -1

        segment_token_ids = self.tokenizer.encode(error_segment, add_special_tokens=False)

        for i in range(len(response_ids) - len(segment_token_ids) + 1):
            window = response_ids[i : i + len(segment_token_ids)]
            if window == segment_token_ids:
                return i, i + len(segment_token_ids)

        logging.warning(f"CalLearner: Could not find exact token match for segment: '{error_segment}'")
        return -1, -1

    def compute_advantages_with_cal(self, trajectory: Dict) -> torch.Tensor:
        """
        This is our new, core advantage calculation logic.
        It creates a sparse advantage tensor based on the CAL's output.
        """
        device = torch.cuda.current_device()
        advantages = []
        
        # In the learner, 'info' is a list of dictionaries, one for each item in the batch.
        # The key is likely 'cal_outputs' if coming from CALOracle, or other keys if from MATHOracle.
        # Let's handle both cases gracefully.
        oracle_infos = trajectory.get("info", [])

        for i in range(len(trajectory["input_ids"])):
            response_ids = trajectory["response_ids"][i]
            per_token_advantage = torch.zeros(len(response_ids), device=device)
            
            error_segment = oracle_infos[i].get("cal_error_segment", "")

            if error_segment:
                start_tok_idx, end_tok_idx = self._map_segment_to_token_indices(response_ids, error_segment)

                if start_tok_idx != -1:
                    num_error_tokens = max(1, end_tok_idx - start_tok_idx)
                    penalty_per_token = self.args.negative_reward / num_error_tokens
                    per_token_advantage[start_tok_idx:end_tok_idx] = penalty_per_token
            
            advantages.append(per_token_advantage)

        return torch.nn.utils.rnn.pad_sequence(advantages, batch_first=True, padding_value=0.0).to(device)


    def learning_step(self, trajectory: Dict) -> Dict:
        """
        This is the main override of the PPO learning step. It orchestrates the
        entire process for a single batch of data using our CAL-based advantages.
        """
        args: CALArgs = self.args
        device = torch.cuda.current_device()

        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        logps = torch.tensor(trajectory["response_logprobs"]).to(device)
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, trajectory["prompt_ids_lens"])
        response_masks = completion_masks[:, 1:]

        advantages = self.compute_advantages_with_cal(trajectory)

        stats = defaultdict(list)
        for _ in range(args.num_ppo_epochs):
            batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(input_ids), args.train_batch_size_per_device):
                mini_batch_inds = batch_inds[b_st : b_st + args.train_batch_size_per_device]

                mb_advantage = advantages[mini_batch_inds]
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_loss_masks = loss_masks[mini_batch_inds]

                logits = self.model(mb_input_ids, attention_mask=mb_att_mask)["logits"].float()
                new_logps = self.get_batch_logps(logits, mb_input_ids, mb_response_masks)

                logprobs_diff = new_logps - mb_logps
                ratio = torch.exp(logprobs_diff)
                pg_losses = -mb_advantage * ratio
                pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                
                pg_loss_max = torch.max(pg_losses, pg_losses2)
                
                pg_loss = self.masked_aggregator(pg_loss_max, mb_response_masks, axis=1)
                loss = (pg_loss * mb_loss_masks).mean()

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
        
        infos = {
            "train/cal_loss": loss.detach().cpu().item(),
            "train/advantage_mean": advantages.mean().cpu().item(),
            "train/advantage_std": advantages.std().cpu().item(),
        }
        return infos

def run_cal_rl(args: CALArgs):
    program, local_resources = get_program(
        args, learner_cls=CalLearner, actor_cls=CalActor
    )
    logging.info(f"--- [Main, PID: {os.getpid()}] Program created. Handing off to launchpad... ---")
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )
    logging.info(f"--- [Main, PID: {os.getpid()}] launchpad has returned. Program finished. ---")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"--- [Main, PID: {os.getpid()}] Script started. Parsing arguments... ---")
    args: CALArgs = get_default_args(CALArgs)
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.
    args = default_args_validation(args)
    logging.info(f"--- [Main, PID: {os.getpid()}] Arguments parsed. Calling run_cal_rl... ---")
    run_cal_rl(args)

"""

python oat/experiment/cal.py \
    --gpus 1 \
    --launch_type local_mt \
    --collocate \
    --vllm_sleep \
    --pretrain EleutherAI/pythia-160m \
    --prompt_data lkevinzc/gsm8k \
    --train_split "train" \
    --input_key question \
    --output_key answer \
    --max_train 128 \
    --use_cal_oracle \
    --prompt_template qwen_math \
    --cal_model_name "gemini-1.5-flash-latest" \
    --cal_few_shot_path "scripts/cal_few_shot_examples.json" \
    --num_prompt_epoch 1 \
    --max_sgd_steps 2 \
    --rollout_batch_size 16 \
    --train_batch_size_per_device 4

python oat/experiment/cal.py \
    --gpus 1 \
    --launch_type local_mt \
    --collocate \
    --vllm_sleep \
    --pretrain EleutherAI/pythia-160m \
    --prompt_data lkevinzc/gsm8k \
    --train_split "train" \
    --input_key question \
    --output_key answer \
    --max_train 128 \
    --use_cal_oracle \
    --prompt_template qwen_math \
    --cal_model_name "gemini-1.5-flash-latest" \
    --cal_few_shot_path "scripts/cal_few_shot_examples.json" \
    --num_prompt_epoch 1 \
    --max_sgd_steps 2 \
    --rollout_batch_size 16 \
    --train_batch_size_per_device 4

python oat/experiment/cal.py \
    --gpus 1 \
    --pretrain "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --critic_pretrain "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --prompt_data "lkevinzc/gsm8k" \
    --train_split "train" \
    --eval_data "lkevinzc/gsm8k" \
    --eval_split "test" \
    --use_cal_oracle=False \
    ... # other training parameters for a full run



python oat/experiment/cal.py \
    --gpus 1 \
    --pretrain "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --prompt_data "lkevinzc/gsm8k" \
    ...
    --use_cal_oracle=True \
    --cal_model_name "gemini-1.5-flash-latest" \
    --cal_few_shot_path "scripts/cal_few_shot_examples.json" \
    ...


"""