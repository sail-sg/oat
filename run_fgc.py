print("--- RUNNING FINAL, SELF-CONTAINED FGC EXPERIMENT SCRIPT ---")

import logging
import torch
import os
import json
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import tree

# --- Step 1: Import all necessary components from the OAT library ---
from oat.algorithms.ppo import PPOArgs, PPOLearner
from oat.actors.reward import RewardActor
from oat.args import get_default_args
from oat.interface import get_program, lp
from oat.types import TrajectoryData, Metric
from oat.oracles.base import RewardOracleBase
from oat.utils.data import PromptDataset
from datasets import load_dataset

# --- Step 2: Define our custom arguments ---
@dataclass
class FGCArgs(PPOArgs):
    """Our custom arguments, inheriting from PPOArgs."""
    cal_model_name: str = field(default="gemini-1.5-flash-latest", metadata={"help": "The Gemini model for the Credit Assignment LLM."})
    cal_few_shot_path: str = field(default="cal_few_shot_examples.json", metadata={"help": "Path to few-shot examples for the CAL."})
    negative_reward: float = field(default=-1.0, metadata={"help": "The negative reward value assigned to the error segment."})
    prompt_data_name: str = field(default="main", metadata={"help": "The 'name' or 'config' of the dataset on Hugging Face."})

# --- Step 3: Define our CALOracle  ---
class CALOracle(RewardOracleBase):
    def __init__(self, cal_model_name: str, few_shot_path: str, api_key_env: str = "GOOGLE_API_KEY"):
        api_key = os.getenv(api_key_env)
        if not api_key: raise ValueError(f"'{api_key_env}' not set. Please export your GOOGLE_API_KEY.")
        genai.configure(api_key=api_key)
        
        generation_config = {"temperature": 0.0, "max_output_tokens": 150}
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        
        self.model = genai.GenerativeModel(model_name=cal_model_name, generation_config=generation_config, safety_settings=safety_settings)
        
        with open(few_shot_path, 'r') as f: self.few_shot_examples = json.load(f)
        self.system_prompt_text = self._build_system_prompt_text()
        logging.info(f"CAL Oracle: Initialized with Gemini model '{cal_model_name}'.")

    def _build_system_prompt_text(self) -> str:
        prompt_parts = [
            "You are a meticulous Credit Assignment LLM (CAL). Your task is to analyze a 'Correct Solution' and an 'Incorrect Solution' to a given 'Question'. You must identify the single sentence in the 'Incorrect Solution' that represents the first logical and substantial divergence from the correct reasoning path.",
            "Your output must be ONLY the exact divergent sentence text from the 'Incorrect Solution' and nothing else. Do not add explanations or introductory phrases.",
            "\nHere are some examples of the task:"
        ]
        for ex in self.few_shot_examples:
            prompt_parts.extend(["\n---\n", f"Question: {ex['question']}", f"Correct Solution: {ex['correct_solution']}", f"Incorrect Solution: {ex['incorrect_solution']}", f"Error Segment: {ex['error_segment']}"])
        prompt_parts.extend(["\n---\n", "Now, perform this task for the following input. Only provide the 'Error Segment'."])
        return "\n".join(prompt_parts)

    def get_error_segment(self, question: str, correct_solution: str, incorrect_solution: str) -> str:
        user_prompt = f"Question: {question}\nCorrect Solution: {correct_solution}\nIncorrect Solution: {incorrect_solution}\nError Segment:"
        try:
            response = self.model.generate_content([self.system_prompt_text, user_prompt])
            return response.text.strip()
        except Exception as e:
            logging.error(f"CAL Oracle API Error: {e}")
            return ""

    def get_reward(self, inputs: List[str], responses: List[str], references: List[str], **kwargs) -> Tuple[torch.Tensor, Metric]:
        cal_outputs = []
        for i in range(len(inputs)):
            error_segment = self.get_error_segment(question=inputs[i], correct_solution=references[i], incorrect_solution=responses[i])
            cal_outputs.append({"error_segment": error_segment})
        dummy_rewards = torch.zeros(len(inputs))
        return dummy_rewards, {"cal_outputs": cal_outputs}

# --- 4. DEFINE OUR OWN FIXED VERSION OF get_datasets ---
def get_datasets_fixed(args, tokenizer, strategy):
    prompt_dataset_from_hub = load_dataset(args.prompt_data, name=args.prompt_data_name)
    train_dataset = PromptDataset(prompt_dataset_from_hub[args.train_split], tokenizer, strategy, input_key=args.input_key, output_key=args.output_key)
    eval_dataset = None
    if args.eval_split and args.eval_split in prompt_dataset_from_hub:
        eval_dataset = PromptDataset(prompt_dataset_from_hub[args.eval_split], tokenizer, strategy, input_key=args.input_key, output_key=args.output_key)
    return train_dataset, eval_dataset

# --- Step 5: Define our custom Actor ---
class FGCActor(RewardActor):
    """
    Our custom actor for Fine-Grained Credit Assignment.
    """
    def __init__(self, args: FGCArgs, policy, ref_policy, **kwargs):
        """
        It accepts the 3 known positional arguments
        and collects all others (like ipc_server) into kwargs.
        """
        # Step 1: Extract the keyword arguments that the parent class needs
        # from the kwargs dictionary.
        ipc_server = kwargs.get('ipc_server')
        vllm_args = kwargs.get('vllm_args')

        # Step 2: Initialize the parent class (ActorBase) with the arguments
        # it requires. This resolves all previous TypeErrors.
        super().__init__(args=args, ipc_server=ipc_server, vllm_args=vllm_args)

        # Step 3: Manually assign the policy models, which are given to us
        # as positional arguments but not handled by the parent.
        self.policy = policy
        self.ref_policy = ref_policy

        # Step 4: Now that the actor is fully constructed, initialize our
        # custom CALOracle component.
        logging.info(f"FGCActor: Initializing CALOracle with model {args.cal_model_name}")
        self.cal_oracle = CALOracle(args.cal_model_name, args.cal_few_shot_path)
    # --- The step method ---
    def step(self, prompts: List[str], formatted_prompts: List[str], references: List[str] = None) -> List[TrajectoryData]:
        logging.info(f"FGCActor received {len(prompts)} prompts. Generating {self.args.num_samples} rollouts per prompt...")
        outputs = self.generate(formatted_prompts, self.sampling_params)
        all_candidates = [output.text.strip() for prompt_output in outputs for output in prompt_output.outputs]

        num_generations = len(all_candidates)
        repeated_prompts = [p for p in prompts for _ in range(self.args.num_samples)]
        repeated_references = [r for r in references for _ in range(self.args.num_samples)]

        _, oracle_info = self.cal_oracle.get_reward(repeated_prompts, all_candidates, repeated_references)

        info_dict = {"actor/generate_time": 0}
        info_dict.update({f"oracle/{k}": v for k, v in oracle_info.items()})

        trajectory_data = []
        gen_idx = 0
        for i in range(len(outputs)):
            for j in range(len(outputs[i].ßoutputs)):
                output_obj = outputs[i].outputs[j]
                traj = TrajectoryData(
                    prompt=repeated_prompts[gen_idx],
                    prompt_ids=outputs[i].prompt_token_ids,
                    response=output_obj.text.strip(),
                    response_ids=output_obj.token_ids,
                    response_logprobs=[l[t].logprob for t,l in zip(output_obj.token_ids, output_obj.logprobs)],
                    rewards=[0.0] * len(output_obj.token_ids),
                    loss_mask=True,
                    info=info_dict,
                )
                trajectory_data.append(traj)
                gen_idx += 1

        logging.info(f"FGCActor finished step, sending {len(trajectory_data)} trajectories.")
        return self.ipc_client.serialize_ipc(trajectory_data)

# --- Step 6: Define our custom Learner  ---
class FCGLearner(PPOLearner):
    def prepare_data(self, strategy, tokenizer):
        self.prompts_dataset, self.eval_prompts_dataset = get_datasets_fixed(self.args, tokenizer, strategy)
        self.prompts_dataloader = strategy.setup_dataloader(self.prompts_dataset, self.args.rollout_batch_size_per_device)

    def _map_segment_to_token_indices(self, generation_ids: List[int], generation_text: str, error_segment: str) -> tuple[int, int]:
        if not error_segment: return -1, -1
        char_start = generation_text.find(error_segment)
        if char_start == -1: return -1, -1
        char_end = char_start + len(error_segment)
        
        try:
            offsets = self.tokenizer(generation_text, return_offsets_mapping=True)["offset_mapping"]
        except:
            logging.warning("Tokenizer does not support offset mapping, cannot apply fine-grained reward.")
            return -1, -1

        start_tok, end_tok = -1, -1
        for i, (start_char, end_char) in enumerate(offsets):
            if start_char == 0 and end_char == 0: continue
            if start_tok == -1 and start_char >= char_start: start_tok = i
            if start_char < char_end: end_tok = i + 1
            
        if start_tok == -1 or end_tok <= start_tok: return -1, -1
        return start_tok, min(end_tok, len(generation_ids))

    def learning_step(self, trajectory: Dict) -> Dict:
        metrics_dict = trajectory.get("info", {})
        if "oracle" in metrics_dict and "cal_outputs" in metrics_dict["oracle"]:
            logging.info("CAL outputs detected. Applying fine-grained rewards.")
            per_token_rewards_list = []
            
            generations_text = trajectory["response"] 
            generation_ids_list = trajectory["response_ids"]
            cal_outputs = metrics_dict["oracle"]["cal_outputs"]
            max_len = max(len(ids) for ids in generation_ids_list)

            for i in range(len(generations_text)):
                reward_tensor = torch.zeros(max_len, dtype=torch.float32)
                error_segment = cal_outputs[i].get("error_segment", "") if i < len(cal_outputs) else ""
                
                start_tok, end_tok = self._map_segment_to_token_indices(generation_ids_list[i], generations_text[i], error_segment)

                if start_tok != -1:
                    num_error_tokens = max(1, end_tok - start_tok)
                    per_token_penalty = self.args.negative_reward / num_error_tokens
                    reward_tensor[start_tok:end_tok] = per_token_penalty
                    if i % 10 == 0: logging.info(f"Applied penalty to segment: '{error_segment}'")
                
                per_token_rewards_list.append(reward_tensor)
            
            # This is the crucial step: we replace the placeholder rewards in the trajectory
            # with our new fine-grained, per-token reward tensors.
            trajectory["rewards"] = per_token_rewards_list

        # Now, we call the original PPOLearner's learning_step with our MODIFIED trajectory.
        # This allows us to reuse all the complex PPO logic without rewriting it.
        return super().learning_step(trajectory)

# --- Step 7: Define the main run function ---
def run_fgc(args: FGCArgs):
    program, local_resources = get_program(args, learner_cls=FCGLearner, actor_cls=FGCActor)
    lp.launch(program, launch_type=args.launch_type, local_resources=local_resources)

if __name__ == "__main__":
    args: FGCArgs = get_default_args(FGCArgs)
    args.algo = "FGC_PPO"
    run_fgc(args)
    