#!/bin/bash
# Build flash-attn 2.8.3.post1 for sm_120 ONLY.  Lessons baked in:
#  1. setup.py hardcodes -gencode and IGNORES TORCH_CUDA_ARCH_LIST.
#  2. It appends "-gencode" and "arch=..." as SEPARATE list items, so disabling
#     only the arch lines leaves dangling "-gencode" flags ->
#     "nvcc fatal: '-gencode' is not in 'keyword=value' format".
#     Patch the single consumption site instead.
#  3. MAX_JOBS x NVCC_THREADS must fit the step's 24 CPUs (20x4=80 thrashed).
#  4. uv venvs have no pip; use uv, and reuse uv's already-unpacked sdist.
set -x
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/sya225/uv_cache
export MAX_JOBS=12
export NVCC_THREADS=2
V=/home/sya225/opd/envs/verl_opd
SRC_CACHE=$(ls -d /localscratch/sya225/uv_cache/sdists-v9/pypi/flash-attn/2.8.3.post1/*/src 2>/dev/null | head -1)
WORK=/localscratch/sya225/fa_build

echo "=== [1/4] stage pristine source  $(date -Is) ==="
[ -d "$SRC_CACHE" ] || { echo NO_CACHED_SDIST; exit 1; }
rm -rf "$WORK"; mkdir -p "$WORK"
cp -a "$SRC_CACHE" "$WORK/src"
rm -rf "$WORK/src/build"
cd "$WORK/src" || exit 1

echo "=== [2/4] patch the cc_flag consumption site -> sm_120 only  $(date -Is) ==="
python3 - <<'PY'
import pathlib, sys
p = pathlib.Path("setup.py"); s = p.read_text()
old = '"nvcc": append_nvcc_threads(nvcc_flags + cc_flag),'
new = '"nvcc": append_nvcc_threads(nvcc_flags + ["-gencode", "arch=compute_120,code=sm_120"]),'
if old not in s:
    print("PATCH_TARGET_NOT_FOUND"); sys.exit(1)
s = s.replace(old, new)
p.write_text(s)
print("patched ->", new.strip())
PY
[ $? -eq 0 ] || { echo PATCH_FAILED; exit 1; }

echo "=== [3/4] build+install  $(date -Is) ==="
uv pip install --python "$V/bin/python" . --no-build-isolation --no-deps > /localscratch/sya225/logs/fa5_build.log 2>&1
echo "install_rc=$?"
tail -4 /localscratch/sya225/logs/fa5_build.log

echo "=== [4/4] verify (from /tmp so we import the INSTALLED pkg, not ./flash_attn)  $(date -Is) ==="
cd /tmp
"$V/bin/python" -c "
import torch, flash_attn
print('flash_attn', flash_attn.__version__, 'at', flash_attn.__file__)
from flash_attn import flash_attn_func
q = torch.randn(1,128,8,64, device='cuda', dtype=torch.bfloat16)
o = flash_attn_func(q,q,q,causal=True)
print('sm_120 kernel OK', tuple(o.shape), 'finite', torch.isfinite(o).all().item())
from flash_attn.bert_padding import unpad_input, pad_input
print('bert_padding OK (real package)')
import transformers.utils as tu
print('is_flash_attn_2_available:', tu.is_flash_attn_2_available())
"
echo "FA_BUILD_DONE exit=$?  $(date -Is)"
