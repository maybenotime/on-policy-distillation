#!/bin/bash
#SBATCH --job-name=analysis-ood-all5
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# OOD recovery-rate analysis — all 5 (student, teacher) pairs
# Runs sequentially on one L40S. Each pair ~1–2h; total ~8h within budget.
# Output: results/ood_prompts_math_with_answers_correct_100_<student>_<teacher>.pkl

source ~/.bashrc
cd /home/sya225/on-policy-distillation
git pull
source test_env/bin/activate
mkdir -p logs results

PROMPTS=data/prompts_math_with_answers_correct_100.jsonl
NUM_ROLLOUTS=10
N_POSITIONS=10
UTIL=0.8
BS=2

run_ood() {
    local STUDENT=$1
    local TEACHER=$2
    local STUDENT_TAG=$(basename "$STUDENT")
    local TEACHER_TAG=$(basename "$TEACHER")
    local OUT="results/ood_${STUDENT_TAG}_${TEACHER_TAG}.pkl"
    echo ""
    echo "========================================"
    echo "OOD: $STUDENT_TAG ← $TEACHER_TAG"
    echo "========================================"
    python source/ood_analysis.py \
        --student_model "$STUDENT" \
        --teacher_model "$TEACHER" \
        --prompts_file "$PROMPTS" \
        --output_path "$OUT" \
        --num_rollouts $NUM_ROLLOUTS \
        --num_sample_positions $N_POSITIONS \
        --batch_size $BS \
        --use_vllm \
        --vllm_gpu_memory_utilization $UTIL
    echo "Saved: $OUT"
}

run_ood Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct
run_ood Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-7B-Instruct
run_ood Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-14B-Instruct
run_ood Qwen/Qwen2.5-3B-Instruct   Qwen/Qwen2.5-7B-Instruct
run_ood Qwen/Qwen2.5-3B-Instruct   Qwen/Qwen2.5-14B-Instruct

echo ""
echo "All OOD analyses done."
ls -lh results/ood_*.pkl
