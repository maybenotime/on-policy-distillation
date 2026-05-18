#!/bin/bash -l
#SBATCH -J bench-eval-fir
#SBATCH --account=rrg-angelx
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:30:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Benchmark eval on Fir (single H100). Same pattern as Vulcan version.
#
# Required env:
#   RUN_DIR    — training output dir with checkpoint-* subdirs
# Optional env:
#   IS_QWEN3   — "1" => use eval_math_reasoning_qwen3.sh + ENABLE_THINKING=false

set -e

if [ -z "${RUN_DIR:-}" ]; then echo "ERROR: RUN_DIR not set" >&2; exit 1; fi

IS_QWEN3="${IS_QWEN3:-0}"

REPO_DIR=/scratch/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

mkdir -p logs

export HF_HOME=/scratch/sya225/hf_cache
export TRANSFORMERS_CACHE=/scratch/sya225/hf_cache
export MODELSCOPE_CACHE=/tmp/sya225_eval_ms_$SLURM_JOB_ID
export TORCH_HOME=/scratch/sya225/hf_cache/torch
export VLLM_CACHE_ROOT=/scratch/sya225/hf_cache/vllm
export TMPDIR=/tmp/sya225_eval_tmp_$SLURM_JOB_ID
export EVAL_OUTPUT_BASE=/scratch/sya225/eval_output
export HF_DATASETS_CACHE=/tmp/sya225_eval_ds_$SLURM_JOB_ID
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$MODELSCOPE_CACHE/lockers" "$TORCH_HOME" "$EVAL_OUTPUT_BASE" "$HF_DATASETS_CACHE"

RUN_TAG=$(basename "$RUN_DIR")

FINAL_CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
if [ -z "$FINAL_CKPT" ]; then
    echo "ERROR: no checkpoint found under $RUN_DIR" >&2
    exit 1
fi

echo "================================================================"
echo "[bench-eval-fir] RUN_DIR=$RUN_DIR"
echo "[bench-eval-fir] FINAL_CKPT=$FINAL_CKPT"
echo "[bench-eval-fir] IS_QWEN3=$IS_QWEN3"
echo "================================================================"

source eval_test/bin/activate
if [ "$IS_QWEN3" = "1" ]; then
    ENABLE_THINKING=false bash scripts/eval_math_reasoning_qwen3.sh "$FINAL_CKPT"
else
    bash scripts/eval_math_reasoning.sh "$FINAL_CKPT"
fi

echo "[bench-eval-fir] ALL DONE for $RUN_TAG"
