#!/bin/bash
# Download Qwen3-8B (teacher) + Qwen3-4B (student) for code OPD.
# Both are the HYBRID Qwen3 models whose chat template supports enable_thinking;
# Qwen3-8B-Instruct-2507 does not exist, so this pairing is the only way to get
# a symmetric non-thinking teacher/student prompt format.
set -x
export HF_HOME=/localscratch/USER/hf_cache
export PATH=$HOME/.local/bin:$PATH
V=/home/USER/opd/envs/verl_opd
for M in Qwen/Qwen3-8B Qwen/Qwen3-4B; do
  echo "=== downloading $M  $(date -Is) ==="
  "$V/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$M', allow_patterns=['*.json','*.safetensors','*.txt','*.model'], max_workers=8)
print('OK', p)
"
done
echo "=== sizes ==="
du -sh /localscratch/USER/hf_cache/hub/models--Qwen--Qwen3-8B /localscratch/USER/hf_cache/hub/models--Qwen--Qwen3-4B 2>/dev/null
echo "DL_DONE=$? $(date -Is)"
