#!/usr/bin/env bash
set -euo pipefail

# Run from the scripts/ dir so external_plugins paths stored in checkpoint
# args.json (e.g. `../source/prefix_curriculum_plugin.py` — these were saved
# relative to the train-time cwd which was scripts/) resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# Route through swift_eval_patched.py — Vulcan compute nodes can reach
# huggingface.co but NOT modelscope.cn (Squid 403). modelscope SDK has no
# offline mode (it always calls repo_exists which hits modelscope.cn), so
# pre-downloading to MODELSCOPE_CACHE doesn't help. Stay on HF hub.
# Patcher also monkey-patches AMCAdapter.record_to_sample to .get()
# year/url since AI-MO/aimo-validation-amc lacks those fields.
# Per-job unique port for internal vLLM server to avoid same-node collisions
# (multiple bench jobs landing on the same compute node otherwise share port
# 8000 and the OpenAI client hits the wrong server's model_list).
EVAL_PORT="${EVAL_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 30000)))}"

CUDA_VISIBLE_DEVICES=0 \
python "$SCRIPT_DIR/../source/swift_eval_patched.py" \
    --use_hf true \
    $EVAL_ARGS \
    --eval_backend Native \
    --infer_backend vllm \
    --port "$EVAL_PORT" \
    --eval_dataset ${EVAL_DATASETS:-gsm8k aime24 aime25 amc math_500 hmmt_feb_2026} \
    --eval_output_dir "${EVAL_OUTPUT_BASE:-/localscratch/sya225/eval_output}/${MODEL_TAG}" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 16384 \
    --eval_num_proc 100
