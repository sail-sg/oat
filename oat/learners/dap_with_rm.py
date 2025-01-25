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

from typing import Any, Dict, List

import torch
import torch.distributed as dist

from oat.learners.dap import DAPLearner
from oat.rm import model
from oat.types import PreferenceData, RewardData
from oat.utils.buffer import UniformBuffer
from oat.utils.distributed import torch_type_codec


class DAPwRMLearner(DAPLearner):
    """Collocated DAP and reward model learning."""

    def _init(self, args, actors) -> None:
        super()._init(args, actors)
        self.rm = None
        self.learn_rm_only = args.learn_rm_only
        self.fixed_reg = args.rm_fixed_reg
        self.train_budget = args.rm_train_budget

        assert args.exp_method != "no" and args.rm_pretrain == ""
        rm_cls = getattr(model, args.exp_method)
        if self.strategy.is_rank_0():
            self.rm: model.RewardModel = rm_cls(args).to(torch.cuda.current_device())
            self.r_buffer = UniformBuffer(args.r_buffer_maxlen)
        self.train_rm_info = rm_cls.get_metrics()

    def process_preference_data(self, data_list: List[PreferenceData], raw_prompts):
        super().process_preference_data(data_list, raw_prompts)
        c_feats = torch.stack([data.chosen_feature for data in data_list]).unsqueeze(
            dim=1
        )
        r_feats = torch.stack([data.rejected_feature for data in data_list]).unsqueeze(
            dim=1
        )
        pair_feats = torch.cat([c_feats, r_feats], dim=1).to(
            torch.cuda.current_device()
        )  # (micro_b, 2, d)
        same_masks = torch.tensor([data.same for data in data_list]).to(
            torch.cuda.current_device()
        )  # (micro_b,)
        model_data_masks = torch.tensor([data.is_model_data for data in data_list]).to(
            torch.cuda.current_device()
        )  # (micro_b,)

        all_pair_feats = self.strategy.gather(pair_feats)
        all_same_masks = self.strategy.gather(same_masks)
        all_model_data_masks = self.strategy.gather(model_data_masks)
        if self.rm:
            self.r_buffer.extend(
                RewardData(
                    pair_features=all_pair_feats,
                    loss_masks=1 - (all_same_masks | all_model_data_masks).float(),
                )
            )

    def learn(self, learning_round):
        train_info = {}
        # NOTE Put reward learning after policy learning otherwise program gets stuck.
        if not self.learn_rm_only:
            train_info.update(super().learn(learning_round))
        train_info.update(self._reward_learning())
        return train_info

    def get_misc_info(self) -> Dict[str, Any]:
        info = super().get_misc_info()
        r_buffer_len = 0
        if self.rm:
            r_buffer_len = self.r_buffer.size
        info.update({"r_buffer_len": self.strategy.all_reduce(r_buffer_len, "max")})
        return info

    def sync_params_to_actors(self):
        """Additionally sync reward model params."""
        # Sync RM.
        if self.rm:
            for name, param in self.rm.named_parameters():
                shape = param.shape
                futs = [
                    actor.futures.update_rm(
                        name,
                        dtype=torch_type_codec(param.dtype),
                        shape=shape,
                    )
                    for actor in self.actors
                ]
                dist.broadcast(param.data, 0, group=self._model_update_group)
                _ = [fut.result() for fut in futs]

        dist.barrier()

        if not self.learn_rm_only:
            # Sync policy.
            super().sync_params_to_actors()

    def _reward_learning(self):
        total_num_queries = self.strategy.all_reduce(self.query_step, "sum")
        if self.rm and total_num_queries < self.train_budget:
            if self.fixed_reg:
                total_num_queries = self.rm.train_bs
            self.r_buffer.total_num_queries = total_num_queries
            train_rm_info = self.rm.learn(self.r_buffer)
            assert self.train_rm_info.keys() == train_rm_info.keys()
            self.train_rm_info = train_rm_info
        dist.barrier()
        self.strategy.broadcast(self.train_rm_info)
        return self.train_rm_info
