#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}}"

export HF_HOME=$SLURM_TMPDIR/.cache/huggingface
export TORCH_HOME=$SLURM_TMPDIR/.cache/torch  
export VLLM_CACHE_ROOT=$SLURM_TMPDIR/.cache/vllm
export TMPDIR=$SLURM_TMPDIR/tmp
mkdir -p $SLURM_TMPDIR/.cache/huggingface $SLURM_TMPDIR/.cache/torch $SLURM_TMPDIR/tmp $SLURM_TMPDIR/.cache/vllm


CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --use_hf true \
    --model "${MODEL}" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k aime24 aime25 amc math_500 \

