#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=rsft-opsd-3b-14b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=4:30:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Chapter 5: RSFT → OPSD-14B — 3B student, RSFT warm-start initialization.
# RSFT first eliminates preference bias (student aligned to its own correct rollouts),
# then OPSD eliminates recovery failure (14B privileged context).
# Orthogonal two-mechanism sequential intervention.

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"

RSFT_CKPT="${BASE}/output/rsft_Qwen2.5-3B-Instruct/v0-20260508-033345/checkpoint-200"

echo "========================================"
echo "RSFT → OPSD 3B — 14B ref"
echo "RSFT checkpoint: ${RSFT_CKPT}"
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
export OPSD_TEACHER_MODEL=Qwen2.5-14B-Instruct
export OPSD_DATA_PATH="${BASE}/data/math_all_teachers_correct.jsonl"
export DATA_SIZE=840
export SFT_ALPHA=0.0
export OUTPUT="${BASE}/output/rsft_then_opsd_3B_14bref"

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}_rsft_opsd_3b.txt"
        echo "Checkpoint: $CHECKPOINT"
    fi
}
trap save_checkpoint EXIT

cd "${BASE}/scripts"
VLLM_UTIL=0.35 bash opsd.sh

echo "Done: RSFT → OPSD 3B 14B-ref"
