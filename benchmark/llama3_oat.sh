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

# 6.9B: [Config 2] Collocate actors and oracle servers.
## Actor: 4 vLLM instances each running on 1 GPU (0~3); 
## Learner: DeepSpeed zero-2 over 4 GPUs (4~7); 
## Oracle: 4 parallel RM workers each running on 1 GPU (0~3).
# 3.a) Start Mosec RM service.
python -m oat.oracles.remote.server --remote-rm-model RLHFlow/ArmoRM-Llama3-8B-v0.1
# 3.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash-attn \
    --ref-offload \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --vllm-gpu-ratio 0.55 \
    --zero-stage 3 \
    --dap-algo DPO \
    --beta 0.01 \
    --preference-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
    --remote-rm-client-workers 4 \
    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --apply-chat-template \
    --prompt-data HuggingFaceH4/ultrafeedback_binarized \
    --train-split train_prefs \
    --prompt-max-length 1800 \
    --input-key prompt \
    --output-key chosen \
    --sync-params-every 10 \
    --max-train 51200 \
    --temperature 0.8 \
    --top-p 0.95 \
    --generate-max-length 1024 \
    --eval-data lkevinzc/alpaca_eval2 \
    --eval-split eval \
    --eval-input-key instruction \
    --eval-output-key meta-llama/Meta-Llama-3-8B-Instruct \
    --eval-generate-max-length 2048 \
    --eval-temperature 0.9 \
    --eval-top-p 1 \
    --train-batch-size 128 \
    --rollout-batch-size 128 \
    --rollout-batch-size-per-device 32 \
    --pi-buffer-maxlen-per-device 32 \
    --train-batch-size-per-device 1 \
    --eval-steps 40 \
    --use-wb \
    --wb-project oat-benchmark \
    --wb-run-name llama3-8b


# 6.9B: [Config 3] Collocate actors and oracle servers + Asynchronous execution.
## Actor: 4 vLLM instances each running on 1 GPU (0~3); 
## Learner: DeepSpeed zero-2 over 4 GPUs (4~7); 
## Oracle: 4 parallel RM workers each running on 1 GPU (0~3).
### By asynchronous, the data collected is not exactly on-policy; but DPO should
### robust to minor data off-policyness.
# 3.a) Start Mosec RM service.
python -m oat.oracles.remote.server --remote-rm-model RLHFlow/ArmoRM-Llama3-8B-v0.1
# 3.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash-attn \
    --ref-offload \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --asynchronous \
    --vllm-gpu-ratio 0.55 \
    --zero-stage 3 \
    --dap-algo DPO \
    --beta 0.01 \
    --preference-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
    --remote-rm-client-workers 4 \
    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --apply-chat-template \
    --prompt-data HuggingFaceH4/ultrafeedback_binarized \
    --train-split train_prefs \
    --prompt-max-length 1800 \
    --input-key prompt \
    --output-key chosen \
    --sync-params-every 10 \
    --max-train 51200 \
    --temperature 0.8 \
    --top-p 0.95 \
    --generate-max-length 1024 \
    --eval-data lkevinzc/alpaca_eval2 \
    --eval-split eval \
    --eval-input-key instruction \
    --eval-output-key meta-llama/Meta-Llama-3-8B-Instruct \
    --eval-generate-max-length 2048 \
    --eval-temperature 0.9 \
    --eval-top-p 1 \
    --train-batch-size 128 \
    --rollout-batch-size 128 \
    --rollout-batch-size-per-device 32 \
    --pi-buffer-maxlen-per-device 32 \
    --train-batch-size-per-device 1 \
    --eval-steps 40 \
    --use-wb \
    --wb-project oat-benchmark \
    --wb-run-name llama3-8b_async
