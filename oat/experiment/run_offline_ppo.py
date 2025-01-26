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


from oat.algorithms.ppo import OfflinePPOLearner, PPOArgs
from oat.args import default_args_validation, get_default_args


def run_ppo(args):
    cls = OfflinePPOLearner

    def __init__(self, args):
        # Hack to discard DistributedLauncher and use deepspeed launcher.
        self.args = args
        self.actors = []
        self.ipc_server = None

    cls.__init__ = __init__
    learner = cls(args=args)
    learner.run()


if __name__ == "__main__":
    args = get_default_args(PPOArgs)

    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.
    args.reward_key = "final_reward"  # Debugging purpose.

    args = default_args_validation(args)
    run_ppo(args)
