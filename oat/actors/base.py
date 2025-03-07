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

import abc
import logging
import time
from typing import List, Union

import torch
import tree
import vllm

from oat import oracles
from oat.args import OATArgs
from oat.rm import model
from oat.types import PreferenceData, TrajectoryData
from oat.utils.distributed import WorkerWrap, torch_type_codec
from oat.utils.ipc import PlasmaShmClient

logging.getLogger("vllm").setLevel(logging.ERROR)


class ActorBase(abc.ABC):
    """Actor handles the interaction between the agent and the environment."""

    def __init__(self, ipc_server, vllm_args, args: OATArgs) -> None:
        self.args = args
        self.eval_mode = False
        self.generate_mode = False

        # Measuring the **online** performance
        self.enable_online_evaluation = args.online_evaluation

        self.ipc_client = PlasmaShmClient(ipc_server)

        # ###################################
        # ####      vLLM Generation      ####
        # ###################################
        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            n=args.num_samples,
        )

        self.__vllm_version__ = vllm.__version__

        assert self.__vllm_version__ >= "0.7.2", "Upgrade to vLLM >= 0.7.2"

        vllm_args.update({"seed": time.time_ns() % 2**32, "worker_cls": WorkerWrap})
        for _ in range(5):
            try:
                self.llm = vllm.LLM(**vllm_args)
            except Exception as e:
                # In case of timeout.
                time.sleep(5)
                logging.warning(f"{e}")
                logging.warning("Re-trying...")
        self.tokenizer = self.llm.get_tokenizer()
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        # ###################################
        # ####     Feedback Oracles      ####
        # ###################################
        oracle_cls = oracles.get_cls(args.oracle)
        logging.info(f"Using reward oracle {args.oracle} {oracle_cls}")
        self.oracle = (
            oracle_cls(
                reward_model_path=args.oracle,
                tokenizer_path=args.pretrain,
                remote_rm_url=args.remote_rm_url,  # Only for remote RM.
                max_workers=args.remote_rm_client_workers,  # Only for remote RM.
            )
            if oracle_cls is not None
            else None
        )
        self.oracle_batch_size = args.oracle_batch_size

    def generate(self, prompts: List[str], sampling_params: vllm.SamplingParams):
        self.generate_mode = True
        if self.tokenizer.bos_token:
            # lstrip bos_token because vllm will add it.
            prompts = [p.lstrip(self.tokenizer.bos_token) for p in prompts]
        outputs = self.llm.generate(
            prompts, sampling_params=sampling_params, use_tqdm=False
        )
        if self.tokenizer.bos_token:
            # make sure vllm added bos_token.
            assert self.tokenizer.bos_token_id in outputs[0].prompt_token_ids

        self.generate_mode = False
        return outputs

    @abc.abstractmethod
    def generate_and_maybe_eval(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ):
        """
        1) Generate responses for given prompts;
        2) Optionally evaluate the win rate over references based on the oracle reward model.
        """

    @abc.abstractmethod
    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[Union[PreferenceData, TrajectoryData]]:
        """Step the actor.

        Given a prompt x, K responses {y_1, ..., y_K} are sample from the behavior LLM pi_beta,
        from which some responses are selected to query the oracle for feedback signal.
        The final constructed pair (x, y_w, y_l) is inserted into the replay buffer for learners.

        Args:
            prompts: A list of prompt texts.
            formatted_prompts: A list of chat template formatted prompt texts.
            references: A list of reference texts.
        """

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        self._model_update_group = self.llm.collective_rpc(
            "init_process_group",
            args=(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            ),
        )

    def is_generating(self):
        return self.generate_mode

    def update_weight(
        self, name, dtype, shape, cuda_ipc_handles=None, empty_cache=False
    ):
        return self.llm.collective_rpc(
            "update_weight", args=(name, dtype, shape, cuda_ipc_handles, empty_cache)
        )

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        """Sleep & Wake Up.
        sleep & wake_up are used together to offload model weights & kv cache to CPUs then onload.
        They are particularly useful when actors & learners collocate.
        """
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def update_rm(self, name, dtype, shape):
        assert self.learning_rm
        dtype = torch_type_codec(dtype)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)
        params_dict = dict(self.explorer.reward_model.named_parameters())
        model.default_weight_loader(params_dict[name], weight)
        del weight

    def notify_eval_start(self, eval=True):
        """Temporarily cache the current behavior policy weights to CPU."""
        if eval:
            self.eval_mode = True
        logging.debug("Start offloading...")
        st = time.time()
        self.cache_model_state = tree.map_structure(
            lambda x: x.cpu(), self.model.state_dict()
        )
        logging.debug(f"Finished offloading in {time.time() - st} seconds")

    def notify_eval_done(self, eval=True):
        """Load cached behavior policy weights to GPU."""
        if eval:
            assert self.eval_mode
        logging.debug("Start loading from cpu...")
        st = time.time()
        self.model.load_state_dict(self.cache_model_state)
        if self.args.enable_prefix_caching:
            self.reset_prefix_cache()
        logging.debug(f"Finished loading in {time.time() - st} seconds")
        if eval:
            self.eval_mode = False
