#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}}"

CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --use_hf true \
    --model "${MODEL}" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k aime24 aime25 amc math_500 \

