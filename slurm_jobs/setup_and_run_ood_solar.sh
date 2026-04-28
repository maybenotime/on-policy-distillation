#!/bin/bash
#SBATCH -J ood-analysis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc

REPO_DIR=/home/sya225/on-policy-distillation
REPO_URL=git@github.com:maybenotime/on-policy-distillation.git

# Clone repo if not present
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull

# Create virtualenv if not present
if [ ! -d "test_env" ]; then
    echo "Creating virtualenv..."
    python3 -m venv test_env
    source test_env/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source test_env/bin/activate
fi

mkdir -p logs

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-14B-Instruct"}
NUM_ROLLOUTS=${3:-10}
PROMPTS_FILE=${4:-"data/prompts_math_with_answers_correct_100.jsonl"}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")
DATASET_TAG=$(basename "$PROMPTS_FILE" .jsonl | sed 's/^ood_//')

python source/ood_analysis.py \
    --student_model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --prompts_file "$PROMPTS_FILE" \
    --output_path "results/ood_${DATASET_TAG}_${STUDENT_TAG}_${TEACHER_TAG}.pkl" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --num_sample_positions 10 \
    --batch_size 2 \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.8
