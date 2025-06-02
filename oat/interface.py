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
"""Defining how components interface with each other."""
import logging
from typing import Type

import launchpad as lp
import torch
from launchpad.nodes.python import local_multi_processing

from oat.actors import PreferenceActor
from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.learners.base import LearnerBase
from oat.utils.ipc import PlasmaShmServer
from oat.utils.launcher import get_free_port


def get_program(
    args: OATArgs,
    learner_cls: Type[LearnerBase],
    actor_cls: Type[ActorBase] = PreferenceActor,
):
    """Define the default distributed program topology with configs."""
    program = lp.Program("oat")
    gpu_offset = (args.group_rank * args.gpus) % torch.cuda.device_count()
    # Resource.
    if args.collocate:
        actor_gpus = learner_gpus = list(range(args.gpus))
    else:
        if args.gpus % 2 == 0:
            actor_gpus = list(range(args.gpus // 2))
            learner_gpus = list(range(args.gpus // 2, args.gpus))
        else:
            logging.warning(
                "Number of GPUs not divisible by 2, one GPU will be forced to collocate learner and actor."
            )
            actor_gpus = list(range(args.gpus // 2 + 1))
            learner_gpus = list(range(args.gpus // 2, args.gpus))
    actor_gpus = [gpu + gpu_offset for gpu in actor_gpus]
    learner_gpus = [gpu + gpu_offset for gpu in learner_gpus]

    learner_world_size = len(learner_gpus) * args.num_groups
    args.learner_gpus_per_group = len(learner_gpus)

    logging.warning(
        f"=== GPU allocations ===\nActor: {actor_gpus}, Learner: {learner_gpus}"
    )

    # IPC.
    ipc_server = program.add_node(
        lp.CourierNode(PlasmaShmServer, size_mb=args.shm_size_mb),
        label=f"ipc_server_{args.group_rank}",
    )

    assert (
        len(actor_gpus) % args.num_gpus_per_actor == 0
    ), "Number of GPUs per actor must be a factor of the number of GPUs in a group."
    assert args.num_gpus_per_actor in [
        1,
        2,
        4,
        8,
    ], "Only 1, 2, 4, 8 GPUs are supported for each actor."

    # Actor.
    vllm_args = {
        "model": args.pretrain,
        "trust_remote_code": True,
        "tensor_parallel_size": args.num_gpus_per_actor,
        "gpu_memory_utilization": args.vllm_gpu_ratio,
        "dtype": "bfloat16",
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_sleep_mode": args.vllm_sleep,
        "max_model_len": args.max_model_len,
    }
    num_actors = len(actor_gpus) // args.num_gpus_per_actor
    actors = []
    local_resources = {}
    for i in range(num_actors):
        label = f"actor_{args.group_rank}_{i}"
        gpus = actor_gpus[
            i * args.num_gpus_per_actor : (i + 1) * args.num_gpus_per_actor
        ]
        actors.append(
            program.add_node(
                lp.CourierNode(actor_cls, ipc_server, vllm_args, args),
                label=label,
            )
        )
        local_resources[label] = local_multi_processing.PythonProcess(
            env={"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in gpus)}
        )
        logging.info(f"Actor {label} launched on GPUs {gpus}")

    # Learner.
    master_addr = args.master_addr
    master_port = args.master_port or get_free_port()
    args.local_rank = 0

    for i in range(0, len(learner_gpus)):
        rank = args.group_rank * len(learner_gpus) + i
        label = f"learner_{args.group_rank}_{i}"
        worker_learner = lp.PyClassNode(
            learner_cls,
            learner_world_size,
            rank,
            i,
            master_addr,
            master_port,
            rank == 0,
            args,
            actors,
            ipc_server,
        )
        program.add_node(worker_learner, label=label)
        local_resources[label] = local_multi_processing.PythonProcess(
            env={"CUDA_VISIBLE_DEVICES": str(learner_gpus[i])}
        )

    return program, local_resources
