#!/bin/bash
# Build flash-attn 2.8.3.post1 for sm_120 ONLY.
#
# Two things that made earlier attempts fail:
#  1. flash-attn's setup.py HARDCODES the -gencode list (~lines 180-191) and
#     ignores TORCH_CUDA_ARCH_LIST, so the arch must be cut by patching setup.py.
#     Without this it builds sm_80/90/100/120 = 4x the kernels.
#  2. MAX_JOBS x NVCC_THREADS must fit the 24 CPUs the srun step has.
#     20 x 4 = 80 thrashed and produced zero .o files in 40 min.
# Also: uv-created venvs have NO pip, so use uv for everything and reuse the
# sdist uv already unpacked into its cache.
set -x
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/USER/uv_cache
export MAX_JOBS=12
export NVCC_THREADS=2
V=/home/USER/opd/envs/verl_opd
SRC_CACHE=$(ls -d /localscratch/USER/uv_cache/sdists-v9/pypi/flash-attn/2.8.3.post1/*/src 2>/dev/null | head -1)
WORK=/localscratch/USER/fa_build

echo "=== [1/4] stage source from uv cache: $SRC_CACHE  $(date -Is) ==="
[ -d "$SRC_CACHE" ] || { echo "NO_CACHED_SDIST"; exit 1; }
rm -rf "$WORK"; mkdir -p "$WORK"
cp -a "$SRC_CACHE" "$WORK/src"
rm -rf "$WORK/src/build"          # drop the half-finished 4-arch objects
cd "$WORK/src" || exit 1

echo "=== [2/4] patch setup.py -> sm_120 only  $(date -Is) ==="
python3 - <<'PY'
import pathlib
p = pathlib.Path("setup.py"); s = p.read_text()
for arch in ("compute_80,code=sm_80", "compute_90,code=sm_90", "compute_100,code=sm_100"):
    s = s.replace(f'cc_flag.append("arch={arch}")', f'pass  # {arch} disabled (sm_120-only build)')
p.write_text(s)
print("gencode lines still active:")
for i, l in enumerate(s.splitlines(), 1):
    if "arch=compute" in l and "pass" not in l:
        print(f"  {i}: {l.strip()}")
PY

echo "=== [3/4] build+install  $(date -Is) ==="
uv pip install --python "$V/bin/python" . --no-build-isolation --no-deps 2>&1 | tail -8

echo "=== [4/4] verify  $(date -Is) ==="
"$V/bin/python" -c "
import torch, flash_attn
print('flash_attn', flash_attn.__version__)
from flash_attn import flash_attn_func
q = torch.randn(1,128,8,64, device='cuda', dtype=torch.bfloat16)
o = flash_attn_func(q,q,q,causal=True)
print('sm_120 kernel OK', tuple(o.shape), 'finite', torch.isfinite(o).all().item())
from flash_attn.bert_padding import unpad_input, pad_input
print('bert_padding OK (real package now)')
import transformers.utils as tu
print('is_flash_attn_2_available:', tu.is_flash_attn_2_available())
"
echo "FA_BUILD_DONE exit=$?  $(date -Is)"
