#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=li-opsd-0p5b
#SBATCH --array=30,50,70,90
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rethink_gap/%x-%A_%a.out
#SBATCH --error=logs/rethink_gap/%x-%A_%a.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Chapter 5: LI-OPSD with 14B reference solutions — 0.5B student.
# Array task ID is the late-intervention ratio in percent:
#   30 / 50 / 70 / 90
#
# This follows the OPSD single-GPU colocate setup. LI-OPSD changes the loss
# region / intervention ratio, not the model footprint, so 1x H100 is enough.

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
RATIO="${SLURM_ARRAY_TASK_ID:-70}"

case "$RATIO" in
    30|50|70|90) ;;
    *) echo "Unsupported LI_OPSD_RATIO=${RATIO}; use 30, 50, 70, or 90." >&2; exit 1 ;;
esac

echo "========================================"
echo "LI-OPSD 0.5B — 14B ref — ratio=${RATIO}%"
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

export MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export OPSD_MODE=single
export OPSD_TEACHER_MODEL=Qwen2.5-14B-Instruct
export OPSD_DATA_PATH="${BASE}/data/math_all_teachers_correct.jsonl"
export DATA_SIZE=840
export SFT_ALPHA=0.0
export LI_OPSD_RATIO="$RATIO"
export OUTPUT="${BASE}/output/li_opsd_0.5B_14bref_ratio${RATIO}"

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}_li_opsd_0p5b_r${RATIO}.txt"
        echo "Checkpoint: $CHECKPOINT"
    fi
}
trap save_checkpoint EXIT

cd "${BASE}/scripts"
VLLM_UTIL=0.40 bash li_opsd.sh

echo "Done: LI-OPSD 0.5B 14B-ref ratio=${RATIO}%"
