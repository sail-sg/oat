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

"""Offline alignment with online vLLM evaluation."""

from dataclasses import dataclass

import launchpad as lp

from oat.actors import PreferenceActor, RewardActor
from oat.args import OATArgs, default_args_validation, get_default_args
from oat.interface import get_program
from oat.learners import OfflineDAPLearner, OfflineSFTLearner
from oat.types import DAPAlgo


@dataclass
class OfflineArgs(OATArgs):
    """Offline DAP from a preference dataset arguments."""

    preference_data: str = ""
    extract_content: bool = False
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    offline_buffer_path: str = "./data/buffer.pkl"


def main(args: OATArgs):
    learner_cls = OfflineDAPLearner if args.algo in DAPAlgo else OfflineSFTLearner
    actor_cls = PreferenceActor if args.oracle_type == "preference" else RewardActor
    program, local_resources = get_program(args, learner_cls, actor_cls)
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args = get_default_args(OfflineArgs)
    args = default_args_validation(args)
    main(args)
