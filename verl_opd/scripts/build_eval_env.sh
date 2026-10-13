#!/bin/bash
# Dedicated CODE-EVAL venv. Reasons it exists rather than reusing anything:
#   * LiveCodeBench/.venv is py3.11 -> vllm 0.27.1 dies at engine startup with
#     "TypeError: type 'array.array' is not subscriptable".
#   * verl_opd is the TRAINING env; it has datasets 4.x (cannot read the
#     LCB v6 arrow cache built by datasets 2.21) and no ninja (vllm's compile
#     backend shells out to it). Installing eval deps there risks the trainer.
#   * evalplus downgrades protobuf 6.33->5.29, which would endanger vllm in any
#     env we care about.
# So: py3.12 + vllm + datasets 2.21 (matches the existing LCB cache) + evalplus
#     + ninja, all in one throwaway env.
set -x
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/sya225/uv_cache
export HF_HOME=/localscratch/sya225/hf_cache
E=/home/sya225/opd/envs/code_eval_env

rm -rf "$E"
uv venv "$E" --python 3.12 || exit 1
uv pip install --python "$E/bin/python" \
    "vllm==0.27.1" "datasets==2.21.0" evalplus ninja pebble 2>&1 | tail -6
echo "INSTALL_RC=$?"

echo "===== version check ====="
"$E/bin/python" - <<'PY'
import shutil
import vllm, datasets, transformers, evalplus
print("vllm        ", vllm.__version__)
print("datasets    ", datasets.__version__)
print("transformers", transformers.__version__)
print("evalplus    ", getattr(evalplus, "__version__", "?"))
print("ninja at    ", shutil.which("ninja"))
PY

echo "===== LCB dataset load (needs datasets 2.21 cache) ====="
cd /home/sya225/opd/LiveCodeBench
HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets \
PYTHONPATH=/home/sya225/opd/LiveCodeBench "$E/bin/python" - <<'PY'
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
d = load_code_generation_dataset(release_version="release_v6", start_date="2025-01-01")
print("LCB v6 problems:", len(d))
PY

echo "===== LCB scoring path import (custom_evaluator chain) ====="
PYTHONPATH=/home/sya225/opd/LiveCodeBench "$E/bin/python" -c "
from lcb_runner.runner.scenario_router import build_prompt_benchmark, get_metrics
print('scenario_router import OK')
" 2>&1 | tail -3

echo "===== HumanEval+ data ====="
"$E/bin/python" -c "
from evalplus.data import get_human_eval_plus
print('HumanEval+ problems:', len(get_human_eval_plus()))
" 2>&1 | tail -2
echo "EVAL_ENV_DONE $(date -Is)"
