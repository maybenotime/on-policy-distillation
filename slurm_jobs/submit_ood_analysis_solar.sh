#!/bin/bash
#SBATCH -J ood-analysis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodelist=cs-venus-15
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc
cd /home/sya225/on-policy-distillation
source test_env/bin/activate

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-3B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-7B-Instruct"}
NUM_ROLLOUTS=${3:-10}
PROMPTS_FILE=${4:-"data/prompts_math_with_answers_correct_100.jsonl"}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

python source/ood_analysis.py \
    --student_model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --prompts_file "$PROMPTS_FILE" \
    --output_path "results/ood_analysis_${STUDENT_TAG}_${TEACHER_TAG}.pkl" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --num_sample_positions 10 \
    --batch_size 2 \
    --use_vllm
