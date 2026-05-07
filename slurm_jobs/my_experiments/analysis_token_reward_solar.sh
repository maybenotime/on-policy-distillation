#!/bin/bash
#SBATCH --job-name=analysis-token-reward-all5
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Token-level reward analysis — all 5 (student, teacher) pairs
# Runs sequentially on one L40S. Each pair ~1h; total ~5h within budget.
# Output: results/token_rewards_<student>_<teacher>.pkl

source ~/.bashrc
cd /home/sya225/on-policy-distillation
git pull
source test_env/bin/activate
mkdir -p logs results

PROMPTS=data/prompts_math_with_answers_correct_100.jsonl

run_token_reward() {
    local STUDENT=$1
    local TEACHER=$2
    local STUDENT_TAG=$(basename "$STUDENT")
    local TEACHER_TAG=$(basename "$TEACHER")
    local OUT="results/token_rewards_${STUDENT_TAG}_${TEACHER_TAG}.pkl"
    echo ""
    echo "========================================"
    echo "Token reward: $STUDENT_TAG ← $TEACHER_TAG"
    echo "========================================"
    python source/token_reward_analysis.py \
        --student_model "$STUDENT" \
        --teacher_models "$TEACHER" \
        --prompts_file "$PROMPTS" \
        --output_path "$OUT" \
        --max_new_tokens 8192 \
        --temperature 1.0 \
        --batch_size 2
    echo "Saved: $OUT"
}

run_token_reward Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct
run_token_reward Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-7B-Instruct
run_token_reward Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-14B-Instruct
run_token_reward Qwen/Qwen2.5-3B-Instruct   Qwen/Qwen2.5-7B-Instruct
run_token_reward Qwen/Qwen2.5-3B-Instruct   Qwen/Qwen2.5-14B-Instruct

echo ""
echo "All token reward analyses done."
ls -lh results/token_rewards_*.pkl
