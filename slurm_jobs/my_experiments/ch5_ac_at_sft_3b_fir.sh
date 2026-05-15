#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=ac-at-sft-3b
#SBATCH --array=10,30,50,70,90
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rethink_gap/%x-%A_%a.out
#SBATCH --error=logs/rethink_gap/%x-%A_%a.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Chapter 5 AC-AT-SFT for 3B student, ratios 10/30/50/70/90.
# Requires data from ac_at_data_gen_3b_first5k_fir.sh.
#
# Submit after data generation:
#   sbatch --dependency=afterok:<DATA_GEN_JOBID> ac_at_sft_3b_first5k_fir.sh

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_TAG="${MODEL##*/}"

AT_RATIO="${AT_RATIO:-${SLURM_ARRAY_TASK_ID:-50}}"
case "$AT_RATIO" in
    10|30|50|70|90) ;;
    *)
        echo "AT_RATIO must be one of 10, 30, 50, 70, 90; got '${AT_RATIO}'" >&2
        exit 1
        ;;
esac

AC_AT_DATA_DIR="${AC_AT_DATA_DIR:-${BASE}/data/chapter5-ac-at-3b-first5k}"
AC_AT_DATA_PATH="${AC_AT_DATA_PATH:-${AC_AT_DATA_DIR}/hybrid_ac_at_3b_${AT_RATIO}_first5k.jsonl}"
OUTPUT="${OUTPUT:-${BASE}/output/ac_at_sft_${MODEL_TAG}_ratio${AT_RATIO}_data5000}"

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

if [ ! -f "$AC_AT_DATA_PATH" ]; then
    echo "AC-AT data file not found: $AC_AT_DATA_PATH" >&2
    echo "Run ac_at_data_gen_3b_first5k_fir.sh first." >&2
    exit 1
fi

DATA_SIZE=$(wc -l < "$AC_AT_DATA_PATH")

echo "========================================"
echo "Chapter 5 AC-AT-SFT (3B)"
echo "========================================"
echo "BASE              = $BASE"
echo "MODEL             = $MODEL"
echo "AT_RATIO          = $AT_RATIO"
echo "AC_AT_DATA_PATH   = $AC_AT_DATA_PATH"
echo "DATA_SIZE         = $DATA_SIZE"
echo "OUTPUT            = $OUTPUT"

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}_ac_at_3b_${AT_RATIO}.txt"
        echo "Checkpoint: $CHECKPOINT"
    else
        echo "No checkpoint found" >&2
    fi
}
trap save_checkpoint EXIT

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model "$MODEL" \
    --use_hf true \
    --dataset "$AC_AT_DATA_PATH" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --save_steps 80 \
    --save_total_limit 6 \
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

echo "AC-AT-SFT (3B) done: ratio=${AT_RATIO}"
