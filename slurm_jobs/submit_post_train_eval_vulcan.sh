#!/bin/bash -l
#SBATCH -J pinned-eval
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:59:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Pinned eval ONLY — recovery rate / per-token PPL / entropy on 5 ckpts.
# Benchmark (gsm8k/math_500/etc.) is now a separate job submitted in parallel
# (see submit_benchmark_eval_vulcan.sh) so they don't serialize.
# Single L40S 48GB. Chained after training via --dependency=afterok:<TRAIN_JOBID>.
#
# Required env (export via --export=ALL,KEY=VAL,... at sbatch time):
#   RUN_DIR    — training output dir containing checkpoint-* subdirs
#   TEACHER    — teacher HF id (e.g. Qwen/Qwen2.5-14B-Instruct or Qwen/Qwen3-14B)
# Optional env:
#   IS_QWEN3   — "1" => use ENABLE_THINKING=false (forced)

set -e

if [ -z "${RUN_DIR:-}" ]; then echo "ERROR: RUN_DIR not set" >&2; exit 1; fi
if [ -z "${TEACHER:-}" ]; then echo "ERROR: TEACHER not set" >&2; exit 1; fi

IS_QWEN3="${IS_QWEN3:-0}"

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

mkdir -p logs results/pinned_curves

export HF_HOME=/scratch/sya225/hf_cache
export HF_DATASETS_CACHE=/scratch/sya225/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/sya225/hf_cache
export MODELSCOPE_CACHE=/scratch/sya225/hf_cache/modelscope
export TORCH_HOME=/scratch/sya225/hf_cache/torch
export VLLM_CACHE_ROOT=/scratch/sya225/hf_cache/vllm
export TMPDIR=/scratch/sya225/hf_cache/tmp
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$MODELSCOPE_CACHE" "$TORCH_HOME"

RUN_TAG=$(basename "$RUN_DIR")

echo "================================================================"
echo "[pinned-eval] RUN_DIR=$RUN_DIR"
echo "[pinned-eval] TEACHER=$TEACHER"
echo "[pinned-eval] IS_QWEN3=$IS_QWEN3"
echo "================================================================"

if [ "$IS_QWEN3" = "1" ]; then
    PINNED_THINKING=false
else
    PINNED_THINKING=none
fi

source test/bin/activate
ENABLE_THINKING="$PINNED_THINKING" \
bash scripts/run_pinned_eval.sh "$RUN_DIR" "$TEACHER" data/pinned_eval_64.jsonl

echo "[pinned-eval] ALL DONE for $RUN_TAG"
