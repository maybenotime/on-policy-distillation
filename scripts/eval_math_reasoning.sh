#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root so external_plugins paths (e.g. source/opsd_plugin.py)
# stored in checkpoint args.json resolve correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}}"
MODEL_TAG="${MODEL##*/}"

export HF_HOME=/home/sya225/.cache/huggingface
export MODELSCOPE_CACHE=/home/sya225/.cache/modelscope
export TORCH_HOME=/home/sya225/.cache/torch
export VLLM_CACHE_ROOT=/home/sya225/.cache/vllm
export TMPDIR=/home/sya225/.cache/tmp
export HF_DATASETS_CACHE=/home/sya225/.cache/huggingface/datasets

mkdir -p $HF_HOME $MODELSCOPE_CACHE $TORCH_HOME $VLLM_CACHE_ROOT $TMPDIR

CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --use_hf true \
    --model "${MODEL}" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k aime24 aime25 amc math_500 \
    --eval_output_dir eval_output/${MODEL_TAG} \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 16384 \
    --eval_num_proc 100
