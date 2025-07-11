#!/usr/bin/env bash
set -e

echo "remove old environments"
rm -rf build dist *.egg-info **/__pycache__ .oat

python3.10 -m venv .oat && source .oat/bin/activate
python -m pip install -U pip && pip install uv
uv pip install numpy==1.26.4

echo "Torch 2.6.0 + cu124"
uv pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

uv pip install tree transformers accelerate datasets==2.16.0 tokenizers safetensors huggingface-hub
uv pip install hatchling ninja packaging wheel psutil vllm==0.8.4
python3.10 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

export TORCH_CUDA_ARCH_LIST="8.9"      
export MAX_JOBS=${MAX_JOBS:-6}
export FLASH_ATTENTION_SKIP_CUDA_BUILD=0   
UV_CACHE_DISABLED=1 
uv pip install flash-attn==2.7.4.post1 \
       --no-build-isolation \
       --no-binary :all: \
       --no-cache-dir

uv pip install wandb deepspeed einops scipy scikit-learn
uv pip install hatchling editables 

echo "Install repo (+vLLM, launchpad)"
UV_CACHE_DISABLED=1 FLASH_ATTENTION_SKIP_CUDA_BUILD=0 \
uv pip install -e . --no-build-isolation --no-cache-dir

echo "smoke test"
python - <<'PY'
import torch, flash_attn, importlib.metadata as md, numpy as np, sys
print("Torch", torch.__version__, "CUDA", torch.version.cuda)
print("Flash-Attn", flash_attn.__version__)
print("vLLM", md.version("vllm"))
from flash_attn import flash_attn_func
print("Kernel OK", flash_attn_func(
    torch.randn(1,128,8,128,device='cuda',dtype=torch.float16),
    torch.randn(1,128,8,128,device='cuda',dtype=torch.float16),
    torch.randn(1,128,8,128,device='cuda',dtype=torch.float16),
    causal=True
).shape)
PY
echo "Done – activate with:  source .oat/bin/activate"
