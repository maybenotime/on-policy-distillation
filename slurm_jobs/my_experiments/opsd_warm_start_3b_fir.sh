#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=opsd-ws-3b
#SBATCH --array=10,30,50,70,90
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/rethink_gap/%x-%A_%a.out
#SBATCH --error=logs/rethink_gap/%x-%A_%a.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Chapter 5: OPSD warm-started from AT-SFT checkpoint (3B student, 14B teacher context).
# Initializes student from AT-SFT ratio-{10,30,50,70,90} checkpoint instead of base model,
# then runs standard OPSD to test whether style pre-alignment improves distillation.
#
# Usage:
#   sbatch opsd_warm_start_3b_fir.sh
#   # or single ratio: sbatch --array=70 opsd_warm_start_3b_fir.sh

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
AT_RATIO="${SLURM_ARRAY_TASK_ID:-70}"

# ── AT-SFT checkpoint base path ──────────────────────────────────────────────
# 3B AT-SFT checkpoints were saved to a custom path on the cluster.
# Adjust AT_SFT_BASE if the path differs.
AT_SFT_ROOT="${AT_SFT_ROOT:-/3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/3b-ac-at-sft-model}"
AT_SFT_BASE="${AT_SFT_ROOT}/at_sft_Qwen2.5-3B-Instruct_ratio${AT_RATIO}_data5000"
INIT_CHECKPOINT=$(find "$AT_SFT_BASE" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)

if [ -z "$INIT_CHECKPOINT" ]; then
    echo "ERROR: No AT-SFT checkpoint found under $AT_SFT_BASE" >&2
    exit 1
fi

echo "========================================"
echo "OPSD Warm Start — 3B Student"
echo "========================================"
echo "AT_RATIO         = $AT_RATIO"
echo "Init checkpoint  = $INIT_CHECKPOINT"

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

export MODEL="$INIT_CHECKPOINT"
export OPSD_MODE=single
export OPSD_TEACHER_MODEL=Qwen2.5-14B-Instruct
export OPSD_DATA_PATH="${BASE}/data/math_all_teachers_correct.jsonl"
export DATA_SIZE=840
export OUTPUT="${BASE}/output/opsd_ws_3B_atsft_ratio${AT_RATIO}_teacher14B"

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}_ws3b_ratio${AT_RATIO}.txt"
        echo "Checkpoint: $CHECKPOINT"
    else
        echo "No checkpoint found" >&2
    fi
}
trap save_checkpoint EXIT

cd "${BASE}/scripts"
# Lower vLLM utilization for 3B to avoid OOM with colocate mode
VLLM_UTIL=0.35 bash opsd.sh

echo "OPSD warm-start 3B done: ratio=${AT_RATIO}"
