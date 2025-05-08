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

# 1. Start the oracle server.
# `&` runs the server in the background.
python -m oat.oracles.remote.server --cuda-devices 0 &

# 2. Loop until the server is ready (it needs time to load the model and warm up the GPUs.)
PROGRAM="bash /home/aiops/liuzc/oat/k8s/readiness.sh"
while true; do
    echo "Checking remote server..."
    
    # Run the program and check its exit status
    $PROGRAM
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "Server ready!"
        break
    else
        echo "Keep waiting..."
        sleep 10
    fi
done

# 3. Ready to run the experiment.
echo "Start experiment..."
python -m oat.experiment.main \
    --gpus 2 \
    --collocate \
    --algo SimPO \
    --beta 2 \
    --preference-oracle remote \
    --remote-rm-url http://0.0.0.0:8000 \
    --pretrain trl-lib/pythia-1b-deduped-tldr-sft \
    --prompt-data lkevinzc/tldr-with-sft-reference \
    --output_key pythia-1b-reference \
    --sync-params-every 1 \
    --rollout-batch-size-per-device 64 \
    --pi-buffer-maxlen-per-device 64 \
    --train-batch-size-per-device 8 \
    --use-wb \
    --wb-run-name 1b_pairrm_simpo_online

# 4. Clean up.
echo "Stopping the server..."
pkill -f python
