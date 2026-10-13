#!/bin/bash
# 19: download two 1.5B models into the node-local HF cache (no weights under /home)
source /home/sya225/opd/env.sh none; unset HF_HUB_OFFLINE
for m in Skywork/Skywork-OR1-Math-7B; do
  echo "== $m $(date -Is)"
  "$CODE_EVAL_ENV/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$m', allow_patterns=['*.json','*.safetensors','*.txt','*.py','*.model','*.jinja'])
import os; print('->', p); print(sorted(os.listdir(p)))
"
done
du -sh $HF_HOME/hub/models--Skywork--Skywork-OR1-Math-7B
echo "DL19B_DONE $(date -Is)"
