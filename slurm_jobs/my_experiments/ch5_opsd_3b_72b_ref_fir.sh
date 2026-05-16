#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=opsd-3b-72b
#SBATCH --array=0,5
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

# Chapter 5: OPSD with 72B reference solutions — 3B student, base initialization.
# Array task ID encodes sft_alpha × 10:
#   0 → sft_alpha=0.0  (pure OPSD, on-policy KD baseline)
#   5 → sft_alpha=0.5  (OPSD + CE anchor on 72B privileged solutions)
# 72B not loaded at training time; only its pre-computed solutions used as privileged context.

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) SFT_ALPHA="0.0" ;;
    5) SFT_ALPHA="0.5" ;;
    *) echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

ALPHA_TAG=$(echo "$SFT_ALPHA" | tr '.' 'p')

echo "========================================"
echo "OPSD 3B — 72B ref — sft_alpha=${SFT_ALPHA}"
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

export MODEL="Qwen/Qwen2.5-3B-Instruct"
export OPSD_MODE=single
export OPSD_TEACHER_MODEL=Qwen2.5-72B-Instruct
export OPSD_DATA_PATH="${BASE}/data/math_all_teachers_correct.jsonl"
export DATA_SIZE=840
export SFT_ALPHA
export OUTPUT="${BASE}/output/opsd_3B_72bref_alpha${ALPHA_TAG}"

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}_opsd3b_72b_alpha${ALPHA_TAG}.txt"
        echo "Checkpoint: $CHECKPOINT"
    fi
}
trap save_checkpoint EXIT

cd "${BASE}/scripts"
VLLM_UTIL=0.35 bash opsd.sh

echo "Done: 3B OPSD 72B-ref sft_alpha=${SFT_ALPHA}"
