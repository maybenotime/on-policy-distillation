#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root so any external_plugins paths in checkpoint
# args.json resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL="${1:-${MODEL:-Qwen/Qwen3-1.7B}}"
MODEL_TAG="${MODEL##*/}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"

# Caches (overridable; Vulcan sets these to /scratch/sya225/hf_cache).
export HF_HOME="${HF_HOME:-/scratch/sya225/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HF_HOME/vllm}"
export TMPDIR="${TMPDIR:-$HF_HOME/tmp}"
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT"

# Detect LoRA adapter checkpoints.
if [ -f "${MODEL}/adapter_config.json" ] && ! python3 -c "
import json, sys, os
cfg = json.load(open('${MODEL}/config.json')) if os.path.exists('${MODEL}/config.json') else {}
sys.exit(0 if 'model_type' in cfg else 1)
" 2>/dev/null; then
    BASE_MODEL=$(python3 -c "import json; d=json.load(open('${MODEL}/adapter_config.json')); print(d['base_model_name_or_path'])")
    echo "LoRA checkpoint detected. Base model: $BASE_MODEL"
    EVAL_ARGS=(--model "$BASE_MODEL" --adapters "$MODEL")
else
    EVAL_ARGS=(--model "$MODEL")
fi

EVAL_OUT="${EVAL_OUTPUT_BASE:-/scratch/sya225/eval_output}/${MODEL_TAG}_thinking-${ENABLE_THINKING}"
mkdir -p "$EVAL_OUT"

echo "Evaluating $MODEL (enable_thinking=$ENABLE_THINKING)"
echo "Output dir: $EVAL_OUT"

EVAL_DATASETS="${EVAL_DATASETS:-gsm8k aime24 aime25 amc math_500}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4096}"

CUDA_VISIBLE_DEVICES=0 \
python "$SCRIPT_DIR/../source/swift_eval_patched.py" \
    --use_hf true \
    "${EVAL_ARGS[@]}" \
    --enable_thinking "$ENABLE_THINKING" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset $EVAL_DATASETS \
    --eval_output_dir "$EVAL_OUT" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 16384 \
    --eval_num_proc 100 \
    --max_new_tokens "$EVAL_MAX_NEW_TOKENS"
