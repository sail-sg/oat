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

# Reference to https://github.com/OpenRLHF/OpenRLHF.

import errno
import logging
import socket
from datetime import timedelta
from typing import Any, Optional, Union

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm.worker.worker import Worker

# !!! IMPORTANT NOTE !!!(liuzc)
# torch.dtype cannot be passed through lp's rpc due to segmentation fault; use string instead.
_torch_type_decode = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}
_torch_type_encode = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
}


def torch_type_codec(dtype_or_str):
    if isinstance(dtype_or_str, torch.dtype):
        return _torch_type_encode[dtype_or_str]
    elif isinstance(dtype_or_str, str):
        return _torch_type_decode[dtype_or_str]
    else:
        raise ValueError(f"Invalid dtype or str: {dtype_or_str}")


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        pg_options=pg_options,
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    print(
        f"init_process_group: init_method={init_method}, backend={backend}, "
        + f"rank={rank}, world_size={world_size}, group_name={group_name}"
    )
    return pg


class WorkerWrap(Worker):
    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Init torch process group for model weights update"""
        assert (
            torch.distributed.is_initialized()
        ), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logging.info(
            f"init_process_group: master_address={master_address}, master_port={master_port}, "
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )
        return self._model_update_group

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (learner model)"""
        dtype = torch_type_codec(dtype)

        # if torch.distributed.get_rank() == 0:
        #     print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert (
            dtype == self.model_config.dtype
        ), f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        if empty_cache:
            torch.cuda.empty_cache()


def node_ip_address_from_perspective(address: str = "8.8.8.8:53"):
    """IP address by which the local node can be reached *from* the `address`.

    OATArgs:
        address: The IP address and port of any known live service on the
            network you care about.

    Returns:
        The IP address by which the local node can be reached from the address.
    """
    ip_address, port = address.split(":")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This command will raise an exception if there is no internet
        # connection.
        s.connect((ip_address, int(port)))
        node_ip_address = s.getsockname()[0]
    except OSError as e:
        node_ip_address = "127.0.0.1"
        # [Errno 101] Network is unreachable
        if e.errno == errno.ENETUNREACH:
            try:
                # try get node ip address from host name
                host_name = socket.getfqdn(socket.gethostname())
                node_ip_address = socket.gethostbyname(host_name)
            except Exception:
                pass
    finally:
        s.close()

    return node_ip_address
