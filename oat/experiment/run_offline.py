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

"""Offline alignment from fixed dataset."""

from dataclasses import dataclass

from oat.args import OATArgs, default_args_validation, get_default_args
from oat.learners import OfflineDAPLearner, OfflineSFTLearner
from oat.types import DAPAlgo


@dataclass
class OfflineArgs(OATArgs):
    """Offline DAP from a preference dataset arguments."""

    preference_data: str = "HuggingFaceH4/ultrafeedback_binarized"
    extract_content: bool = (
        True  # Enable when chosen / reject key contains conversation-style data
    )
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    eval_steps: int = -1


def main(args):
    cls = OfflineDAPLearner if args.algo in DAPAlgo else OfflineSFTLearner

    def __init__(self, args):
        # Hack to discard DistributedLauncher and use deepspeed launcher.
        self.args = args
        self.actors = []
        self.ipc_server = None

    cls.__init__ = __init__
    learner = cls(args=args)
    learner.run()


if __name__ == "__main__":
    args = get_default_args(OfflineArgs)
    args = default_args_validation(args)
    main(args)
