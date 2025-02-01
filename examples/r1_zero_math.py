# Copyright 2025 Garena Online Private Limited
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

import launchpad as lp

from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program
from oat.oracles.countdown import CountdownOracle


class ZeroMathActor(PPOActor):
    def __init__(self, ipc_server, vllm_args, args: PPOArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        if args.oracle == "countdown":
            self.oracle = CountdownOracle()
        else:
            raise NotImplementedError

        # Special treatment to sample from a base model - now only cover Qwen.
        self.sampling_params.stop = (
            [
                "</s>",
                "<|im_end|>",
                "<|endoftext|>",
                "\nUser:",
            ]
            if "qwen" in args.pretrain.lower()
            else []
        )
        self.sampling_params.stop_token_ids = (
            [151645, 151643] if "qwen" in args.pretrain.lower() else []
        )
        self.eval_sampling_params.stop = (
            [
                "</s>",
                "<|im_end|>",
                "<|endoftext|>",
                "\nUser:",
            ]
            if "qwen" in args.pretrain.lower()
            else []
        )
        self.eval_sampling_params.stop_token_ids = (
            [151645, 151643] if "qwen" in args.pretrain.lower() else []
        )


def run_ppo(args: PPOArgs):
    learner_cls = PPOLearner
    actor_cls = ZeroMathActor
    program, local_resources = get_program(args, learner_cls, actor_cls)
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args: PPOArgs = get_default_args(PPOArgs)

    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.

    args = default_args_validation(args)
    run_ppo(args)
