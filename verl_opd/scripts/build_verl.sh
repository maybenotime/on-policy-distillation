#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/sya225/uv_cache
export HF_HOME=/localscratch/sya225/hf_cache
V=/home/sya225/opd/envs/verl_opd
echo "===== install verl[vllm,math] 0.9.0 ====="
time uv pip install --python "$V/bin/python" "verl[vllm,math]==0.9.0" 2>&1 | tail -15
echo "===== EXIT=$? ====="
"$V/bin/python" -c "
import torch, verl, vllm, transformers, tensordict, ray
print('torch      ', torch.__version__, '| cuda', torch.version.cuda)
print('verl       ', verl.__version__)
print('vllm       ', vllm.__version__)
print('transformers', transformers.__version__)
print('tensordict ', tensordict.__version__)
print('ray        ', ray.__version__)
print('arch_list  ', torch.cuda.get_arch_list())
print('sm_120 OK  ', 'sm_120' in torch.cuda.get_arch_list())
"
