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

"""Model inference for using vllm."""

import argparse
import json
import os
import time

from datasets import load_dataset
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="meta-llama/Llama-3.2-1B",
    help="Path to the LLM model",
)
parser.add_argument(
    "--temperature", type=float, default=0.9, help="Temperature for sampling"
)
parser.add_argument(
    "--top_p", type=float, default=1, help="Top-p probability for sampling"
)
parser.add_argument(
    "--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate"
)
parser.add_argument(
    "--output_dir", type=str, default="inference_outputs", help="output_dir"
)
args = parser.parse_args()
args.seed = int(time.time_ns() // 2 * 20)  # Less bias to a fixed random seed.

print(args)

llm = LLM(model=args.model, dtype="bfloat16")


tokenizer = llm.get_tokenizer()

eval_set = load_dataset("lkevinzc/alpaca_eval2")["eval"]

prompts = eval_set["instruction"]

conversations = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]

sampling_params = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    seed=args.seed,
)

if tokenizer.bos_token:
    # removeprefix bos_token because vllm will add it.
    print(conversations[0].startswith(tokenizer.bos_token))
    conversations = [p.removeprefix(tokenizer.bos_token) for p in conversations]

outputs = llm.generate(conversations[:1], sampling_params)

if tokenizer.bos_token:
    # make sure vllm added bos_token.
    assert tokenizer.bos_token_id in outputs[0].prompt_token_ids

outputs = llm.generate(conversations, sampling_params)

# Save the outputs as a JSON file.
output_data = []
model_name = args.model.replace("/", "_")
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    output_data.append(
        {
            "instruction": prompts[i],
            "format_instruction": prompt,
            "output": generated_text,
            "generator": model_name,
        }
    )

output_file = f"{model_name}_{args.seed}.json"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
