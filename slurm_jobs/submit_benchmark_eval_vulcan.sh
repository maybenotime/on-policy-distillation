#!/bin/bash -l
#SBATCH -J bench-eval
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:59:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Benchmark eval ONLY — gsm8k / math_500 / aime24 / aime25 / amc on the final
# checkpoint of a training run. Independent SLURM job so it can run in
# parallel with the pinned-eval job (see submit_post_train_eval_vulcan.sh).
# Single L40S 48GB. Chained after training via --dependency=afterok:<TRAIN_JOBID>.
#
# Required env:
#   RUN_DIR    — training output dir containing checkpoint-* subdirs
# Optional env:
#   IS_QWEN3   — "1" => use eval_math_reasoning_qwen3.sh + ENABLE_THINKING=false

set -e

if [ -z "${RUN_DIR:-}" ]; then echo "ERROR: RUN_DIR not set" >&2; exit 1; fi

IS_QWEN3="${IS_QWEN3:-0}"

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

mkdir -p logs

export HF_HOME=/scratch/sya225/hf_cache
export HF_DATASETS_CACHE=/scratch/sya225/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/sya225/hf_cache
export MODELSCOPE_CACHE=/scratch/sya225/hf_cache/modelscope
export TORCH_HOME=/scratch/sya225/hf_cache/torch
export VLLM_CACHE_ROOT=/scratch/sya225/hf_cache/vllm
export TMPDIR=/scratch/sya225/hf_cache/tmp
export EVAL_OUTPUT_BASE=/scratch/sya225/eval_output
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$EVAL_OUTPUT_BASE"

RUN_TAG=$(basename "$RUN_DIR")

FINAL_CKPT=$(find "$RUN_DIR" -maxdepth 2 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
if [ -z "$FINAL_CKPT" ]; then
    echo "ERROR: no checkpoint found under $RUN_DIR" >&2
    exit 1
fi

echo "================================================================"
echo "[bench-eval] RUN_DIR=$RUN_DIR"
echo "[bench-eval] FINAL_CKPT=$FINAL_CKPT"
echo "[bench-eval] IS_QWEN3=$IS_QWEN3"
echo "================================================================"

source eval_test/bin/activate
if [ "$IS_QWEN3" = "1" ]; then
    ENABLE_THINKING=false bash scripts/eval_math_reasoning_qwen3.sh "$FINAL_CKPT"
else
    bash scripts/eval_math_reasoning.sh "$FINAL_CKPT"
fi

echo "[bench-eval] ALL DONE for $RUN_TAG"
