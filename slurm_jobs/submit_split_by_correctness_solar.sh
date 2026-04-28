#!/bin/bash
#SBATCH -J split-correctness
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:59:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc

REPO_DIR=/home/sya225/on-policy-distillation

if [ ! -d "$REPO_DIR" ]; then
    git clone git@github.com:maybenotime/on-policy-distillation.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull
source test_env/bin/activate

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-3B-Instruct"}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

python source/split_by_correctness.py \
    --student_model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --output_path "data/gsm8k_train_correctness_${STUDENT_TAG}_${TEACHER_TAG}.jsonl" \
    --max_new_tokens 4096 \
    --temperature 1.0 \
    --gpu_memory_utilization 0.85
