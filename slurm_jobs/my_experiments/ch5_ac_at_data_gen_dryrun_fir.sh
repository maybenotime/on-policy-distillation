#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=ac-at-gen-dryrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Dry run for Chapter 5 AC-AT data generation.
# Generates 8 examples for AC-AT-30/50/70.

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
PROMPTS="${PROMPTS:-${BASE}/data/prompts_math_swift.jsonl}"
ANSWERS="${ANSWERS:-${BASE}/data/prompts_math_with_answers.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE}/data/chapter5-ac-at-dryrun}"
LIMIT="${LIMIT:-8}"
STUDENT="${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER="${TEACHER:-Qwen/Qwen2.5-14B-Instruct}"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Chapter 5 AC-AT data generation dry run"
echo "========================================"
echo "BASE       = $BASE"
echo "PROMPTS    = $PROMPTS"
echo "ANSWERS    = $ANSWERS"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "LIMIT      = $LIMIT"
echo "STUDENT    = $STUDENT"
echo "TEACHER    = $TEACHER"

for RATIO in 30 50 70; do
    RATIO_FLOAT="0.${RATIO}"
    OUTPUT="${OUTPUT_DIR}/hybrid_ac_at_${RATIO}_dryrun.jsonl"

    echo "----------------------------------------"
    echo "Dryrun AC-AT-${RATIO}: $OUTPUT"
    echo "----------------------------------------"

    python source/generate_answer_conditioned_hybrid_trajectories.py \
        --prompts "$PROMPTS" \
        --answers "$ANSWERS" \
        --output "$OUTPUT" \
        --student "$STUDENT" \
        --teacher "$TEACHER" \
        --ratio "$RATIO_FLOAT" \
        --limit "$LIMIT"
done

echo "AC-AT dryrun generation finished."
