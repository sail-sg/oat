# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import socket

# Run this file with `torchrun --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$RANK  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT hostfile.py > hostfile`

assert os.environ["LOCAL_RANK"] == "0", "This script should be run with torchrun with --nproc_per_node=1"

torch.distributed.init_process_group(backend="nccl")

rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()

hostname = socket.gethostname()
myline = f"{hostname} slots={torch.cuda.device_count()}"

all_hosts = [None] * world_size

torch.distributed.all_gather_object(all_hosts, myline)

print('\n'.join(all_hosts))

torch.distributed.destroy_process_group()
