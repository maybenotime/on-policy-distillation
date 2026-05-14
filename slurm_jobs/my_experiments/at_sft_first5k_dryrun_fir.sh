#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=at-sft-dryrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --time=00:45:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Dry run for Chapter 5 adaptive-truncation SFT.
# It uses the same 4-GPU launch shape as the full run, but stops after a few
# optimization steps to validate dataset loading, distributed setup, and output.

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_TAG="${MODEL##*/}"

AT_RATIO="${AT_RATIO:-50}"
case "$AT_RATIO" in
    30|50|70) ;;
    *)
        echo "AT_RATIO must be one of 30, 50, 70; got '${AT_RATIO}'" >&2
        exit 1
        ;;
esac

AT_DATA_DIR="${AT_DATA_DIR:-${BASE}/data/chapter5-at-first5k}"
AT_DATA_PATH="${AT_DATA_PATH:-${AT_DATA_DIR}/hybrid_at_${AT_RATIO}_first5k.jsonl}"
MAX_STEPS="${MAX_STEPS:-3}"
OUTPUT="${OUTPUT:-${BASE}/output/dryrun_at_sft_${MODEL_TAG}_ratio${AT_RATIO}}"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export HF_HOME="${HF_HOME:-/home/cta156/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/cta156/.cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-/home/cta156/.cache/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/home/cta156/.cache/vllm}"
export TMPDIR="${TMPDIR:-/home/cta156/.cache/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/cta156/.cache/huggingface/datasets}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$HF_DATASETS_CACHE"

if [ ! -f "$AT_DATA_PATH" ]; then
    echo "AT data file not found: $AT_DATA_PATH" >&2
    echo "Set AT_DATA_DIR or AT_DATA_PATH to the uploaded data location." >&2
    exit 1
fi

DATA_SIZE=$(wc -l < "$AT_DATA_PATH")

echo "========================================"
echo "Chapter 5 AT-SFT dry run"
echo "========================================"
echo "BASE         = $BASE"
echo "MODEL        = $MODEL"
echo "AT_RATIO     = $AT_RATIO"
echo "AT_DATA_PATH = $AT_DATA_PATH"
echo "DATA_SIZE    = $DATA_SIZE"
echo "MAX_STEPS    = $MAX_STEPS"
echo "OUTPUT       = $OUTPUT"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model "$MODEL" \
    --use_hf true \
    --dataset "$AT_DATA_PATH" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --save_steps "$MAX_STEPS" \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_length 4096 \
    --save_only_model true \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --dataset_num_proc 4 \
    --dataloader_num_workers 16 \
    --output_dir "$OUTPUT" \
    --report_to none

echo "AT-SFT dry run done: ratio=${AT_RATIO}"
