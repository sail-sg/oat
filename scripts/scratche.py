# cal_workflow_template.py
"""Skeleton for building a Credit‑Assignment LLM (CAL) workflow on top of the
OAT online‑alignment framework.  The script shows the moving parts you need to
fill in before the actual training run: dataset loading, roll‑out generation,
credit‑assignment with a CAL model, policy‑gradient update, and resource
configuration (ZeRO‑3, FP16, etc.).  Everything is modular so that you can swap
backbones or reward functions with a few lines of code.

This *is not* a turnkey trainer – think of it as a well‑commented template that
highlights the hooks you have to implement.  Search for the keyword TODO to see
places where project‑specific logic is required.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# OAT imports — available once you `pip install oat-llm` or install from source
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner  # type: ignore
from oat.interface import get_program, lp  # type: ignore
from oat.types import TrajectoryData  # type: ignore

###############################################################################
# 0. Command‑line & hyper‑parameters ########################################################
###############################################################################

@dataclass
class CALArgs(PPOArgs):
    """Extension of OAT's PPOArgs with CAL‑specific knobs."""

    # DATA
    dataset_name: str = "hendrycks/math"  # or "yentinglin/aime_2025"
    evaluation_split: str = "test"
    prompt_template: Literal["qwen_math", "r1", "no"] = "qwen_math"

    # CAL MODEL
    cal_backbone: str = "microsoft/deberta-v3-large"
    max_review_len: int = 4096

    # CREDIT‑ASSIGNMENT
    penalty: float = -1.0  # reward to assign to erroneous tokens
    rollout_per_state: int = 4  # K in the VinePPO paper

    # SYSTEM
    dtype: Literal["fp16", "bf16"] = "fp16"
    zero_stage: int = 3  # ZeRO partitioning stage
    teacher_parallel: int = 1  # put CAL on a single GPU


###############################################################################
# 1. Credit‑Assignment model ###################################################
###############################################################################

class CalModel(torch.nn.Module):
    """A very small wrapper that flags *where* two solutions diverge.

    Right now it token‑classifies the concatenation `[wrong] [SEP] [correct]`
    and returns the earliest index where the class=1 ("wrong") logit exceeds
    0.5.  Feel free to swap in anything—from simple diff heuristics to a
    fine‑tuned DeBERTa, or even another causal LLM with span‑prediction tokens.
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.backbone = AutoModelForTokenClassification.from_pretrained(
            backbone_name,
            num_labels=2,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )
        self.eval()  # CAL stays fixed during policy training

    @torch.no_grad()
    def locate_divergence(self, wrong: str, correct: str) -> Tuple[int, int]:
        """Return *(start_idx, end_idx)* (inclusive, exclusive) in `wrong`.
        If no divergence is detected, returns *(len(wrong_tokens)-1, len(wrong_tokens))* so the
        last token gets zeroed by the learner.
        """
        inputs = self.tokenizer(
            wrong,
            correct,
            truncation="only_first",
            return_offsets_mapping=True,
            max_length=self.backbone.config.max_position_embeddings,
            return_tensors="pt",
        )
        logits = self.backbone(**{k: v.to(self.backbone.device) for k, v in inputs.items()}).logits[0]
        probs = torch.sigmoid(logits)[:, 1]  # class‑1 == divergence
        divergent_idxs = (probs > 0.5).nonzero(as_tuple=True)[0]
        if len(divergent_idxs) == 0:
            return (len(inputs["offset_mapping"][0]) - 1, len(inputs["offset_mapping"][0]))
        start = int(divergent_idxs.min())
        end = int(divergent_idxs.max()) + 1
        return start, end


###############################################################################
# 2. Custom Actor ##############################################################
###############################################################################

class MathCalActor(PPOActor):
    """Subclass PPOActor to plug CAL into the usual OAT sampling loop."""

    def __init__(self, ipc_server, vllm_args, args: CALArgs):
        super().__init__(ipc_server, vllm_args, args)
        logging.info("Loading CAL model …")
        self.cal = CalModel(args.cal_backbone).to("cuda:0")  # CAL on a single GPU

    # ---------------------------------------------------------------------
    # ===========  Helper functions  =====================================
    # ---------------------------------------------------------------------

    def _first_correct(self, answers: Sequence[str], references: Sequence[str]) -> int | None:
        """Return index of the first correct answer; None if all wrong."""
        for i, (a, ref) in enumerate(zip(answers, references)):
            # TODO: replace with your task‑specific checker (regex / sympy / etc.)
            if a.strip().split(" ")[-1] == ref.strip():
                return i
        return None

    # ---------------------------------------------------------------------
    # ===========  Core OAT hook  =========================================
    # ---------------------------------------------------------------------

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] | None = None,
    ) -> List[TrajectoryData]:
        assert not self.eval_mode  # we are in training

        # 1. Generate K roll‑outs per prompt
        outputs = self.generate(formatted_prompts, self.sampling_params)

        trajectories: List[TrajectoryData] = []
        for prompt_idx, output in enumerate(outputs):
            all_solutions = [o.text for o in output.outputs]
            # Fallback: if references is None (self‑play), treat *first* solution as provisional correct
            ref_answer = references[prompt_idx] if references else all_solutions[0]
            correct_idx = self._first_correct(all_solutions, [ref_answer] * len(all_solutions))
            if correct_idx is None:
                # No correct solution; choose best‑scoring one to use as pseudo‑reference
                correct_idx = 0
            correct_solution = all_solutions[correct_idx]

            for sol_idx, sol_text in enumerate(all_solutions):
                # 2. Use CAL only if this is an incorrect sample
                if sol_idx == correct_idx:
                    continue  # skip or give neutral reward

                span_start, span_end = self.cal.locate_divergence(sol_text, correct_solution)
                tok_ids = output.outputs[sol_idx].token_ids
                logps = output.outputs[sol_idx].logprobs

                # 3. Build token‑level advantages (= penalty on the erroneous span)
                advantages = torch.zeros(len(tok_ids))
                advantages[span_start:span_end] = self.args.penalty

                # 4. Wrap everything into TrajectoryData expected by PPOLearner
                traj = TrajectoryData(
                    prompt=prompts[prompt_idx],
                    response=sol_text,
                    token_ids=tok_ids,
                    logps=logps,
                    advantages=advantages,
                    ref_answer=ref_answer,
                )
                trajectories.append(traj)
        return trajectories


###############################################################################
# 3. Training driver ###########################################################
###############################################################################

def load_math_dataset(name: str, split: str):
    ds = load_dataset(name, split=split)
    return ds["question"], ds["answer"]


def main():
    args = CALArgs()

    # 1. Prepare data & vLLM arguments ------------------------------------------------------
    questions, answers = load_math_dataset(args.dataset_name, "train")
    vllm_args = {
        "model": args.model_name,
        "trust_remote_code": True,
        "dtype": args.dtype,
    }

    # 2. Launch distributed OAT program -----------------------------------------------------
    program = get_program(MathCalActor, PPOLearner, args, vllm_args, questions, answers)
    lp.launch(program)  # launchpad handles multi‑process orchestration


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
