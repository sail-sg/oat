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

# Dependencies for benchmarking to use *nccl*:
# pip install vllm==0.4.2 transformers==4.43.3 flash-attn==2.5.8 deepspeed==0.14.4

# 1B: [Config 1] Collocate all three workloads.
## Actor: 8 vLLM instances each running on 1 GPU; 
## Learner: DeepSpeed zero-2 over 8 GPUs; 
## Oracle: 8 parallel RM workers each running on 1 GPU.
# 1.a) Start Mosec RM service.
python benchmark/pythia_custom_remote_rm.py --remote-rm-model trl-lib/pythia-1b-deduped-tldr-rm
# 1.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash-attn \
    --rnd-seed \
    --gpus 8 \
    --collocate \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
    --remote-rm-client-workers 8 \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --input-key prompt \
    --output-key pythia-1b-reference \
    --sync-params-every 1 \
    --max-train 2560 \
    --generate-max-length 53 \
    --train-batch-size 128 \
    --rollout-batch-size 128 \
    --rollout-batch-size-per-device 16 \
    --pi-buffer-maxlen-per-device 16 \
    --train-batch-size-per-device 8 \
    --eval-steps 99999 \
    --debug \
    --use-wb \
    --wb-project oat-benchmark \
    --wb-run-name 1b_pythia


# 1B: [Config 2] Collocate actors and oracle servers.
## Actor: 4 vLLM instances each running on 1 GPU (0~3); 
## Learner: DeepSpeed zero-2 over 4 GPUs (4~7); 
## Oracle: 4 parallel RM workers each running on 1 GPU (0~3).
# 1.2.a) Start Mosec RM service.
python benchmark/pythia_custom_remote_rm.py --remote-rm-model trl-lib/pythia-1b-deduped-tldr-rm --cuda-devices 0,1,2,3
# 1.2.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash-attn \
    --rnd-seed \
    --gpus 8 \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
    --remote-rm-client-workers 8 \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --input-key prompt \
    --output-key pythia-1b-reference \
    --sync-params-every 1 \
    --max-train 2560 \
    --generate-max-length 53 \
    --train-batch-size 128 \
    --rollout-batch-size 128 \
    --rollout-batch-size-per-device 32 \
    --pi-buffer-maxlen-per-device 32 \
    --train-batch-size-per-device 8 \
    --eval-steps 99999 \
    --debug \
    --use-wb \
    --wb-project oat-benchmark \
    --wb-run-name 1b_pythia


# 2.8B: [Config 2] Collocate actors and oracle servers.
## Actor: 4 vLLM instances each running on 1 GPU (0~3); 
## Learner: DeepSpeed zero-2 over 4 GPUs (4~7); 
## Oracle: 4 parallel RM workers each running on 1 GPU (0~3).
# 2.a) Start Mosec RM service.
python benchmark/pythia_custom_remote_rm.py --remote-rm-model trl-lib/pythia-2.8b-deduped-tldr-rm --cuda-devices 0,1,2,3
# 2.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash-attn \
    --rnd-seed \
    --gpus 8 \
    --vllm-gpu-ratio 0.35 \
    --zero-stage 2 \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
    --remote-rm-client-workers 8 \
    --pretrain trl-lib/pythia-2.8b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --input-key prompt \
    --output-key pythia-2.8b-reference \
    --sync-params-every 1 \
    --max-train 2560 \
    --generate-max-length 53 \
    --train-batch-size 128 \
    --rollout-batch-size 128 \
    --rollout-batch-size-per-device 32 \
    --pi-buffer-maxlen-per-device 32 \
    --train-batch-size-per-device 2 \
    --eval-steps 99999 \
    --debug \
    --use-wb \
    --wb-project oat-benchmark \
    --wb-run-name 2.8b_pythia


# 6.9B: [Config 2] Collocate actors and oracle servers.
## Actor: 4 vLLM instances each running on 1 GPU (0~3); 
## Learner: DeepSpeed zero-2 over 4 GPUs (4~7); 
## Oracle: 4 parallel RM workers each running on 1 GPU (0~3).
# 3.a) Start Mosec RM service.
python benchmark/pythia_custom_remote_rm.py --remote-rm-model trl-lib/pythia-6.9b-deduped-tldr-rm --tokenizer trl-lib/pythia-6.9b-deduped-tldr-sft --cuda-devices 0,1,2,3
# 3.b) Open another bash and run the experiment.
python -m oat.experiment.main \
    --flash-attn \
    --ref-offload \
    --gradient-checkpointing \
    --rnd-seed \
    --gpus 8 \
    --vllm-gpu-ratio 0.55 \
    --zero-stage 2 \
    --adam-offload \
    --dap-algo DPO \
    --beta 0.1 \
    --preference-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
    --remote-rm-client-workers 8 \
    --pretrain trl-lib/pythia-6.9b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --input-key prompt \
    --output-key pythia-6.9b-reference \
    --sync-params-every 1 \
    --max-train 2560 \
    --generate-max-length 53 \
    --train-batch-size 128 \
    --rollout-batch-size 128 \
    --rollout-batch-size-per-device 32 \
    --pi-buffer-maxlen-per-device 32 \
    --train-batch-size-per-device 4 \
    --eval-steps 99999 \
    --debug \
    --use-wb \
    --wb-project oat-benchmark \
    --wb-run-name 6.9b_pythia
