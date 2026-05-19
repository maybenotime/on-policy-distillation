#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=li-opsd-3b-mix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Chapter 5: LI-OPSD 3B — diagnostic-115 mix50_50 experiment.
# Data: 2500 li_opsd_answer_first5k + 2500 diag-115 (upsampled) = 5000 records.
# LI_OPSD_RATIO=50 fixed (best config from 0.5B sweep).

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"

echo "========================================"
echo "LI-OPSD 3B — diag mix50_50 — ratio=50%"
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
export LI_OPSD_DATA_PATH="${BASE}/data/li_opsd_mix50_50_5k.jsonl"
export DATA_SIZE=5000
export SFT_ALPHA=0.0
export LI_OPSD_RATIO=50
export OUTPUT="${BASE}/output/li_opsd_3B_diagmix50_50_ratio50"
export SLEEP_LEVEL=0
export SAVE_STEPS=25
export SAVE_TOTAL_LIMIT=13

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}_li_opsd_3b_mix50_50.txt"
        echo "Checkpoint: $CHECKPOINT"
    fi
}
trap save_checkpoint EXIT

cd "${BASE}/scripts"
VLLM_UTIL=0.35 bash li_opsd.sh

echo "Done: LI-OPSD 3B diagmix50_50 ratio=50%"
