#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=li-opsd-mix
#SBATCH --array=1,2,3,4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=3:30:00
#SBATCH --output=logs/rethink_gap/%x-%A_%a.out
#SBATCH --error=logs/rethink_gap/%x-%A_%a.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=fc10510,fc10404

# Chapter 5: LI-OPSD 0.5B — diagnostic 115 mixing experiment.
# Baseline: li_opsd_0.5B ratio=50 (pure 5k li_opsd data).
# Here we fix ratio=50 and vary the proportion of diagnostic-115 mixed in:
#   array=1: 90% li_opsd + 10% diag-115  (mix90_10)
#   array=2: 70% li_opsd + 30% diag-115  (mix70_30)
#   array=3: 50% li_opsd + 50% diag-115  (mix50_50)
#   array=4: 30% li_opsd + 70% diag-115  (mix30_70)
# All datasets are 5000 records total (diag-115 upsampled with replacement).

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"

case "${SLURM_ARRAY_TASK_ID}" in
    1) MIX="mix90_10" ;;
    2) MIX="mix70_30" ;;
    3) MIX="mix50_50" ;;
    4) MIX="mix30_70" ;;
    *) echo "Unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

echo "========================================"
echo "LI-OPSD 0.5B — diag mix — ${MIX}"
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
export LI_OPSD_DATA_PATH="${BASE}/data/li_opsd_${MIX}_5k.jsonl"
export DATA_SIZE=5000
export SFT_ALPHA=0.0
export LI_OPSD_RATIO=50
export OUTPUT="${BASE}/output/li_opsd_0.5B_diag${MIX}_ratio50"
export SLEEP_LEVEL=1
export SAVE_STEPS=25
export SAVE_TOTAL_LIMIT=13

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}_li_opsd_${MIX}.txt"
        echo "Checkpoint: $CHECKPOINT"
    fi
}
trap save_checkpoint EXIT

cd "${BASE}/scripts"
VLLM_UTIL=0.30 bash li_opsd.sh

echo "Done: LI-OPSD 0.5B diag-mix ${MIX} ratio=50"
