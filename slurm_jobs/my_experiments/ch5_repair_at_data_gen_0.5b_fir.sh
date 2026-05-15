#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=at-repair-5k
#SBATCH --array=30,50,70
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/rethink_gap/%x-%A_%a.out
#SBATCH --error=logs/rethink_gap/%x-%A_%a.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Chapter 5 teacher-repair generation on 5000 samples for AT-30/50/70.

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
PROMPTS="${PROMPTS:-${BASE}/data/prompts_math_swift.jsonl}"
ANSWERS="${ANSWERS:-${BASE}/data/prompts_math_with_answers.jsonl}"
DRAFT_DIR="${DRAFT_DIR:-${BASE}/data/chapter5-at-first5k}"
OUTPUT_BASE="${OUTPUT_BASE:-${BASE}/data/chapter5-at-repair-sweep-5k}"
LIMIT="${LIMIT:-5000}"
STUDENT="${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-14B-Instruct}"
AT_RATIO="${AT_RATIO:-${SLURM_ARRAY_TASK_ID:-50}}"

case "$AT_RATIO" in
    30|50|70) ;;
    *)
        echo "AT_RATIO must be one of 30, 50, 70; got '${AT_RATIO}'" >&2
        exit 1
        ;;
esac

DRAFT="${DRAFT:-${DRAFT_DIR}/hybrid_at_${AT_RATIO}_first5k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/at${AT_RATIO}}"
RATIO_FLOAT="0.${AT_RATIO}"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Chapter 5 AT-${AT_RATIO} teacher-repair generation (5k)"
echo "========================================"
echo "BASE       = $BASE"
echo "PROMPTS    = $PROMPTS"
echo "ANSWERS    = $ANSWERS"
echo "DRAFT      = $DRAFT"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "LIMIT      = $LIMIT"
echo "TEACHER    = $TEACHER"

python source/chapter5_at50_variant_sweep.py \
    --prompts "$PROMPTS" \
    --answers "$ANSWERS" \
    --draft "$DRAFT" \
    --output_dir "$OUTPUT_DIR" \
    --limit "$LIMIT" \
    --ratio "$RATIO_FLOAT" \
    --ratio_label "$AT_RATIO" \
    --student "$STUDENT" \
    --teacher "$TEACHER" \
    --variants teacher_repair

echo "AT-${AT_RATIO} teacher-repair 5k generation finished."
