#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/sya225/uv_cache
V=/home/sya225/opd/envs/verl_opd
L=/localscratch/sya225/logs
mkdir -p $L

echo "########## [1/3] install flash-attn ##########"
uv pip install --python $V/bin/python flash-attn --no-build-isolation 2>&1 | tail -4

echo "########## [2/3] flash-attn kernel test on sm_120 ##########"
$V/bin/python - <<'PY'
import torch, flash_attn
print("flash_attn", flash_attn.__version__)
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q, causal=True)
print("kernel OK", tuple(o.shape), "finite", torch.isfinite(o).all().item())
PY
echo "FA_TEST_EXIT=$?"

echo "########## [3/3] OPD smoke test ##########"
SMOKE=1 bash /home/sya225/opd/scripts/run_opd_math.sh > $L/smoke6.log 2>&1
echo "SMOKE_EXIT=$?" | tee -a $L/smoke6.log
tail -5 $L/smoke6.log
echo "########## CHAIN_DONE ##########"
