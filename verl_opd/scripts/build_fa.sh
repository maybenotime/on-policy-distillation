#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/USER/uv_cache
export TORCH_CUDA_ARCH_LIST="12.0"     # Blackwell only -> ~4x fewer kernel variants
export MAX_JOBS=20
export NVCC_THREADS=4
V=/home/USER/opd/envs/verl_opd
echo "=== build start $(date -Is)  arch=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS ==="
uv pip install --python $V/bin/python flash-attn --no-build-isolation 2>&1 | tail -6
echo "=== build done $(date -Is) ==="
$V/bin/python -c "
import torch, flash_attn
print('flash_attn', flash_attn.__version__)
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn import flash_attn_func
q = torch.randn(1,128,8,64, device='cuda', dtype=torch.bfloat16)
o = flash_attn_func(q,q,q,causal=True)
print('kernel OK', tuple(o.shape), 'finite', torch.isfinite(o).all().item())
print('bert_padding OK')
"
echo "FA_DONE_EXIT=$?"
