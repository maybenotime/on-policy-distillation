#!/bin/bash -l
#SBATCH -J post-eval
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Post-train eval — pinned eval on ALL checkpoints + benchmark on final checkpoint.
# Single L40S 48GB. Chained after training via --dependency=afterok:<TRAIN_JOBID>.
#
# Required env (export via --export=ALL,KEY=VAL,... at sbatch time):
#   RUN_DIR    — training output dir containing checkpoint-* subdirs
#   TEACHER    — teacher HF id (e.g. Qwen/Qwen2.5-14B-Instruct or Qwen/Qwen3-14B)
# Optional env:
#   IS_QWEN3   — "1" => use eval_math_reasoning_qwen3.sh + ENABLE_THINKING=false (forced)
#   SKIP_PINNED    — "1" => skip pinned eval phase
#   SKIP_BENCHMARK — "1" => skip benchmark phase

set -e

if [ -z "${RUN_DIR:-}" ]; then echo "ERROR: RUN_DIR not set" >&2; exit 1; fi
if [ -z "${TEACHER:-}" ]; then echo "ERROR: TEACHER not set" >&2; exit 1; fi

IS_QWEN3="${IS_QWEN3:-0}"

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

mkdir -p logs results/pinned_curves

# Reuse existing Vulcan cache; both venvs share these.
export HF_HOME=/scratch/sya225/hf_cache
export HF_DATASETS_CACHE=/scratch/sya225/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/sya225/hf_cache
export VLLM_CACHE_ROOT=/scratch/sya225/hf_cache/vllm
export TMPDIR=/scratch/sya225/hf_cache/tmp
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT"

RUN_TAG=$(basename "$RUN_DIR")

echo "================================================================"
echo "[post-eval] RUN_DIR=$RUN_DIR"
echo "[post-eval] TEACHER=$TEACHER"
echo "[post-eval] IS_QWEN3=$IS_QWEN3"
echo "================================================================"

############################################
# Phase 1: pinned eval on all checkpoints  #
############################################
if [ "${SKIP_PINNED:-0}" != "1" ]; then
    echo ""
    echo "===> Phase 1: pinned eval on all checkpoints"
    if [ "$IS_QWEN3" = "1" ]; then
        PINNED_THINKING=false
    else
        PINNED_THINKING=none
    fi

    (
        source test/bin/activate
        ENABLE_THINKING="$PINNED_THINKING" \
        bash scripts/run_pinned_eval.sh "$RUN_DIR" "$TEACHER" data/pinned_eval_64.jsonl
    )
    echo "[post-eval] pinned eval done"
else
    echo "[post-eval] SKIP_PINNED=1, skipping pinned phase"
fi

############################################
# Phase 2: benchmark on final checkpoint   #
############################################
if [ "${SKIP_BENCHMARK:-0}" != "1" ]; then
    echo ""
    echo "===> Phase 2: benchmark on final checkpoint"

    FINAL_CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
    if [ -z "$FINAL_CKPT" ]; then
        echo "ERROR: no checkpoint found under $RUN_DIR" >&2
        exit 1
    fi
    echo "[post-eval] final checkpoint: $FINAL_CKPT"

    (
        source eval_test/bin/activate
        if [ "$IS_QWEN3" = "1" ]; then
            ENABLE_THINKING=false bash scripts/eval_math_reasoning_qwen3.sh "$FINAL_CKPT"
        else
            bash scripts/eval_math_reasoning.sh "$FINAL_CKPT"
        fi
    )
    echo "[post-eval] benchmark done"
else
    echo "[post-eval] SKIP_BENCHMARK=1, skipping benchmark phase"
fi

echo ""
echo "[post-eval] ALL DONE for $RUN_TAG"
