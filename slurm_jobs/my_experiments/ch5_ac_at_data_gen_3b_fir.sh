#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=ac-at-gen-3b
#SBATCH --array=10,30,50,70,90
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rethink_gap/%x-%A_%a.out
#SBATCH --error=logs/rethink_gap/%x-%A_%a.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Chapter 5: Generate 3B AC-AT data for ratios 10/30/50/70/90.
# Student=3B generates prefix, teacher=14B continues with answer conditioning.
# Output: chapter5-ac-at-3b-first5k/hybrid_ac_at_3b_{ratio}_first5k.jsonl

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
STUDENT="${STUDENT:-Qwen/Qwen2.5-3B-Instruct}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-14B-Instruct}"
PROMPTS="${PROMPTS:-${BASE}/data/prompts_math_swift.jsonl}"
ANSWERS="${ANSWERS:-${BASE}/data/prompts_math_with_answers.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE}/data/chapter5-ac-at-3b-first5k}"
LIMIT="${LIMIT:-5000}"
AT_RATIO="${AT_RATIO:-${SLURM_ARRAY_TASK_ID:-50}}"

case "$AT_RATIO" in
    10|30|50|70|90) ;;
    *)
        echo "AT_RATIO must be one of 10, 30, 50, 70, 90; got '${AT_RATIO}'" >&2
        exit 1
        ;;
esac

RATIO_FLOAT="0.${AT_RATIO}"
OUTPUT="${OUTPUT:-${OUTPUT_DIR}/hybrid_ac_at_3b_${AT_RATIO}_first5k.jsonl}"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export HF_HOME="${HF_HOME:-/home/cta156/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/cta156/.cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-/home/cta156/.cache/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/home/cta156/.cache/vllm}"
export TMPDIR="${TMPDIR:-/home/cta156/.cache/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/cta156/.cache/huggingface/datasets}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" \
         "$HF_DATASETS_CACHE" "$OUTPUT_DIR"

echo "========================================"
echo "Chapter 5 3B AC-AT data generation"
echo "========================================"
echo "BASE        = $BASE"
echo "STUDENT     = $STUDENT"
echo "TEACHER     = $TEACHER"
echo "AT_RATIO    = $AT_RATIO"
echo "OUTPUT      = $OUTPUT"
echo "LIMIT       = $LIMIT"

python source/generate_answer_conditioned_hybrid_trajectories.py \
    --prompts  "$PROMPTS" \
    --answers  "$ANSWERS" \
    --output   "$OUTPUT" \
    --student  "$STUDENT" \
    --teacher  "$TEACHER" \
    --ratio    "$RATIO_FLOAT" \
    --limit    "$LIMIT"

echo "3B AC-AT data generation done: ratio=${AT_RATIO}, output=${OUTPUT}"
