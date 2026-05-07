#!/bin/bash
#SBATCH -J token-reward
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclusive
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out

# Parameterized token reward analysis for Solar (L40S / A40)
# Usage: sbatch this_script.sh <student_model> <teacher_model>
# Example:
#   sbatch submit_token_reward_analysis_solar_param.sh \
#       Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-14B-Instruct

source ~/.bashrc
cd /home/sya225/on-policy-distillation
git pull
source test_env/bin/activate
mkdir -p logs results

STUDENT_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER_MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}"
STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

echo "Student: $STUDENT_MODEL"
echo "Teacher: $TEACHER_MODEL"

python source/token_reward_analysis.py \
    --student_model "$STUDENT_MODEL" \
    --teacher_models "$TEACHER_MODEL" \
    --prompts_file data/prompts_math_with_answers_correct_100.jsonl \
    --output_path "results/token_rewards_${STUDENT_TAG}_${TEACHER_TAG}.pkl" \
    --max_new_tokens 8192 \
    --temperature 1.0 \
    --batch_size 2

echo "Done: results/token_rewards_${STUDENT_TAG}_${TEACHER_TAG}.pkl"
