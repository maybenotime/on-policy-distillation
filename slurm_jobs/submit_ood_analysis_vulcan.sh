#!/bin/bash -l
#SBATCH -J qwen3-ood
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# OOD analysis on Vulcan — single L40S 48GB.
# Args:
#   $1 = student model (default Qwen/Qwen3-1.7B)
#   $2 = teacher model (default Qwen/Qwen3-8B)
#   $3 = prompts file (default data/ood_math_100.jsonl)
#   $4 = num rollouts per sample position (default 10)
#   $5 = enable_thinking (true|false|none, default false for Qwen3 non-thinking)

cd /project/aip-angelx/sya225/on-policy-distillation

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

source test/bin/activate

mkdir -p logs results

# Reuse existing /scratch/sya225/hf_cache as HF_HOME (already has Qwen3 models cached).
export HF_HOME=/scratch/sya225/hf_cache
export HF_DATASETS_CACHE=/scratch/sya225/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/sya225/hf_cache
export TMPDIR=/scratch/sya225/hf_cache/tmp
mkdir -p "$TMPDIR"

STUDENT_MODEL=${1:-"Qwen/Qwen3-1.7B"}
TEACHER_MODEL=${2:-"Qwen/Qwen3-8B"}
PROMPTS_FILE=${3:-"data/ood_math_100.jsonl"}
NUM_ROLLOUTS=${4:-10}
ENABLE_THINKING=${5:-"false"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")
DATASET_TAG=$(basename "$PROMPTS_FILE" .jsonl | sed 's/^ood_//')

OUTPUT_PATH="results/ood_${DATASET_TAG}_${STUDENT_TAG}_${TEACHER_TAG}.pkl"
echo "Running OOD analysis: student=$STUDENT_MODEL teacher=$TEACHER_MODEL"
echo "                     dataset=$PROMPTS_FILE  enable_thinking=$ENABLE_THINKING"
echo "                     output=$OUTPUT_PATH"

python source/ood_analysis.py \
    --student_model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --prompts_file "$PROMPTS_FILE" \
    --output_path "$OUTPUT_PATH" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --num_sample_positions 10 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size 2 \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.8 \
    --enable_thinking "$ENABLE_THINKING"
