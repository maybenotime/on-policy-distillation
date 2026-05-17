#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=rsft-opsd-mixed
#SBATCH --array=10,30,50
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=4:30:00
#SBATCH --output=logs/rethink_gap/%x-%A_%a.out
#SBATCH --error=logs/rethink_gap/%x-%A_%a.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Chapter 5: RSFT → OPSD-72B with mixed data (GSM8K + MATH train).
# Array task ID = mix ratio % of base 840 problems:
#   10 → +84  problems  (10% extra from GSM8K+MATH pool)
#   30 → +252 problems  (30% extra)
#   50 → +420 problems  (50% extra)
# All extra problems use human reference solutions as 72B privileged context.

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
RATIO="${SLURM_ARRAY_TASK_ID:-10}"

case "$RATIO" in
    10|30|50) ;;
    *) echo "Unknown SLURM_ARRAY_TASK_ID=${RATIO}" >&2; exit 1 ;;
esac

RSFT_CKPT="${BASE}/output/rsft_Qwen2.5-3B-Instruct/v0-20260508-033345/checkpoint-200"
DATA_PATH="${BASE}/data/mixed_opsd_${RATIO}pct_gsm8k_math.jsonl"
DATA_SIZE=$(wc -l < "$DATA_PATH")

echo "========================================"
echo "RSFT → OPSD 3B — 72B ref — mixed ${RATIO}%"
echo "RSFT checkpoint: ${RSFT_CKPT}"
echo "Data: ${DATA_PATH} (${DATA_SIZE} problems)"
echo "========================================"

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

export MODEL="$RSFT_CKPT"
export OPSD_MODE=single
export OPSD_TEACHER_MODEL=Qwen2.5-72B-Instruct
export OPSD_DATA_PATH="$DATA_PATH"
export DATA_SIZE
export SFT_ALPHA=0.0
export OUTPUT="${BASE}/output/rsft_then_opsd_3B_72bref_mixed${RATIO}pct"

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}_rsft_opsd_mixed${RATIO}.txt"
        echo "Checkpoint: $CHECKPOINT"
    fi
}
trap save_checkpoint EXIT

cd "${BASE}/scripts"
VLLM_UTIL=0.35 bash opsd.sh

echo "Done: RSFT → OPSD 3B 72B-ref mixed ${RATIO}%"
