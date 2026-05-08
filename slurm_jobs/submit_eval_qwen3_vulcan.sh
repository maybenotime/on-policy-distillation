#!/bin/bash -l
#SBATCH -J qwen3-eval
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:59:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Auto-eval after Qwen3 OPD training. Submit with:
#   sbatch --dependency=afterany:<TRAIN_JOB_ID> \
#          --export=ALL,TRAIN_JOB_ID=<TRAIN_JOB_ID>,TRAIN_OUTPUT_BASE=<...> \
#          submit_eval_qwen3_vulcan.sh
#
# Or directly with a known checkpoint path as $1:
#   sbatch submit_eval_qwen3_vulcan.sh <checkpoint_dir>
#
# Required env vars when using dependency mode:
#   TRAIN_JOB_ID         — SLURM job id of upstream training run
#   TRAIN_OUTPUT_BASE    — fallback: training output dir to scan if checkpoint_path file missing
#   ENABLE_THINKING      — optional, default "false" for Qwen3 non-thinking

cd /project/aip-angelx/sya225/on-policy-distillation

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

# Use eval_test venv (has opencompass+evalscope; numpy<2 pinned for compat).
# `swift eval --eval_backend Native` actually triggers opencompass imports.
source eval_test/bin/activate

mkdir -p logs

ENABLE_THINKING="${ENABLE_THINKING:-false}"
export ENABLE_THINKING

# Resolve checkpoint:
#   1. Use $1 if explicitly provided
#   2. Read checkpoint_path_${TRAIN_JOB_ID}.txt
#   3. Fallback: scan TRAIN_OUTPUT_BASE for latest checkpoint-*
CHECKPOINT=""

if [ $# -ge 1 ] && [ -n "$1" ]; then
    CHECKPOINT="$1"
elif [ -n "${TRAIN_JOB_ID:-}" ]; then
    CKPT_FILE="/project/aip-angelx/sya225/on-policy-distillation/checkpoint_path_${TRAIN_JOB_ID}.txt"
    if [ -f "$CKPT_FILE" ] && [ -s "$CKPT_FILE" ]; then
        CHECKPOINT=$(cat "$CKPT_FILE")
    fi
    if [ -z "$CHECKPOINT" ] && [ -n "${TRAIN_OUTPUT_BASE:-}" ]; then
        CHECKPOINT=$(find "$TRAIN_OUTPUT_BASE" -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -1)
    fi
fi

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: no checkpoint resolved" >&2
    echo "  TRAIN_JOB_ID=${TRAIN_JOB_ID:-<unset>}" >&2
    echo "  TRAIN_OUTPUT_BASE=${TRAIN_OUTPUT_BASE:-<unset>}" >&2
    exit 1
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: resolved checkpoint dir does not exist: $CHECKPOINT" >&2
    exit 1
fi

echo "Evaluating checkpoint: $CHECKPOINT"
echo "Enable thinking: $ENABLE_THINKING"

bash scripts/eval_math_reasoning_qwen3.sh "$CHECKPOINT"
