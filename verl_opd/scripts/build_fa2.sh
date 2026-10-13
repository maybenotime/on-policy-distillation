#!/bin/bash
# Build flash-attn for sm_120 ONLY.
# NOTE: flash-attn's setup.py hardcodes the -gencode list (lines ~180-191) and
# ignores TORCH_CUDA_ARCH_LIST entirely, so the arch must be cut by patching
# setup.py. Threads sized to the 24 CPUs the srun step actually has
# (MAX_JOBS x NVCC_THREADS = 12 x 2 = 24); the earlier 20 x 4 = 80 thrashed and
# produced zero object files in 40 minutes.
set -x
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/sya225/uv_cache
export MAX_JOBS=12
export NVCC_THREADS=2
V=/home/sya225/opd/envs/verl_opd
WORK=/localscratch/sya225/fa_build
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"

echo "=== [1/4] fetch sdist $(date -Is) ==="
$V/bin/pip download flash-attn==2.8.3.post1 --no-deps --no-binary :all: -d "$WORK" 2>&1 | tail -3 \
  || uv pip download flash-attn==2.8.3.post1 -d "$WORK" 2>&1 | tail -3
tar xzf flash_attn-*.tar.gz 2>/dev/null || tar xzf flash-attn-*.tar.gz
cd flash*attn-2.8.3.post1

echo "=== [2/4] patch setup.py: drop sm_80/90/100, keep only sm_120 ==="
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("setup.py"); s = p.read_text()
# Neutralise the three non-Blackwell gencode appends, keep compute_120.
for arch in ("compute_80,code=sm_80", "compute_90,code=sm_90", "compute_100,code=sm_100"):
    s = s.replace(f'cc_flag.append("arch={arch}")', f'pass  # arch={arch} disabled')
p.write_text(s)
print("remaining gencode arch lines:")
for i, l in enumerate(s.splitlines(), 1):
    if "arch=compute" in l:
        print(f"  {i}: {l.strip()}")
PY

echo "=== [3/4] build+install $(date -Is) ==="
$V/bin/pip install . --no-build-isolation --no-deps 2>&1 | tail -8
echo "=== [4/4] verify $(date -Is) ==="
$V/bin/python -c "
import torch, flash_attn
print('flash_attn', flash_attn.__version__)
from flash_attn import flash_attn_func
q = torch.randn(1,128,8,64, device='cuda', dtype=torch.bfloat16)
o = flash_attn_func(q,q,q,causal=True)
print('sm_120 kernel OK', tuple(o.shape), 'finite', torch.isfinite(o).all().item())
from flash_attn.bert_padding import unpad_input, pad_input
print('bert_padding OK (real one now)')
"
echo "FA_BUILD_DONE=$?  $(date -Is)"
