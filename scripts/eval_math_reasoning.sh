#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root so external_plugins paths (e.g. source/opsd_plugin.py)
# stored in checkpoint args.json resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}}"
MODEL_TAG="${MODEL##*/}"

export HF_HOME="${HF_HOME:-/home/sya225/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/sya225/.cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-/home/sya225/.cache/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/home/sya225/.cache/vllm}"
export TMPDIR="${TMPDIR:-/home/sya225/.cache/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p $HF_HOME $MODELSCOPE_CACHE $TORCH_HOME $VLLM_CACHE_ROOT $TMPDIR

# Detect LoRA adapter checkpoints (have adapter_config.json but no config.json model_type)
if [ -f "${MODEL}/adapter_config.json" ] && ! python3 -c "
import json, sys
cfg = json.load(open('${MODEL}/config.json')) if __import__('os').path.exists('${MODEL}/config.json') else {}
sys.exit(0 if 'model_type' in cfg else 1)
" 2>/dev/null; then
    BASE_MODEL=$(python3 -c "import json; d=json.load(open('${MODEL}/adapter_config.json')); print(d['base_model_name_or_path'])")
    echo "LoRA checkpoint detected. Base model: $BASE_MODEL"
    EVAL_ARGS="--model ${BASE_MODEL} --adapters ${MODEL}"
else
    EVAL_ARGS="--model ${MODEL}"
fi

# Route through swift_eval_patched.py — needed on Vulcan where Squid blocks
# modelscope.cn (evalscope's TaskConfig defaults to dataset_hub=MODELSCOPE,
# and swift's --use_hf true does NOT propagate to evalscope). Patcher also
# fixes the amc adapter's strict record['year'] access.
CUDA_VISIBLE_DEVICES=0 \
python "$SCRIPT_DIR/../source/swift_eval_patched.py" \
    --use_hf true \
    $EVAL_ARGS \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k aime24 aime25 amc math_500 \
    --eval_output_dir "${EVAL_OUTPUT_BASE:-/localscratch/sya225/eval_output}/${MODEL_TAG}" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 16384 \
    --eval_num_proc 100
