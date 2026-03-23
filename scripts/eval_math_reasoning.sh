#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}}"
MODEL_TAG="${MODEL##*/}"

# export HF_HOME=$SLURM_TMPDIR/.cache/huggingface
# export MODELSCOPE_CACHE=$SLURM_TMPDIR/.cache/modelscope
# export TORCH_HOME=$SLURM_TMPDIR/.cache/torch  
# export VLLM_CACHE_ROOT=$SLURM_TMPDIR/.cache/vllm
# export TMPDIR=$SLURM_TMPDIR/tmp

# mkdir -p $HF_HOME $MODELSCOPE_CACHE $TORCH_HOME $VLLM_CACHE_ROOT $TMPDIR


CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --use_hf true \
    --model "${MODEL}" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k aime24 aime25 amc math_500 \
    --eval_output_dir ../eval_output/${MODEL_TAG} \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 16384 \
    --eval_num_proc 100 \
