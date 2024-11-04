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

from dataclasses import dataclass

import launchpad as lp

from oat.args import OATArgs, default_args_validation, get_default_args
from oat.baselines.xpo import XPOActor, XPOLearner
from oat.interface import get_program


@dataclass
class XPOArgs(OATArgs):
    """Exploratory preference optimization arguments."""

    xpo_alpha: float = 5e-6
    xpo_offload_actor_ref: bool = False


def run_xpo(args):
    program, local_resources = get_program(args, XPOLearner, XPOActor)
    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args = get_default_args(XPOArgs)
    args = default_args_validation(args)
    run_xpo(args)
