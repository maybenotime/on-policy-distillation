#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/sya225/uv_cache
V=/home/sya225/opd/envs/verl_opd
rm -rf "$V"
uv venv "$V" --python 3.12 || exit 1
echo "===== DRY RUN: verl[vllm,math,gpu]==0.9.0 ====="
uv pip install --python "$V/bin/python" "verl[vllm,math,gpu]==0.9.0" --dry-run 2>&1 | tail -60
