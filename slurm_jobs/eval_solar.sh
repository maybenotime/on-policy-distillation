#!/bin/bash
#SBATCH -J on-policy-teacher7b-student3b-5000-eval
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclusive
#SBATCH --time=02:59:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc

cd /home/sya225/on-policy-distillation

source test_env/bin/activate

cd scripts

CHECKPOINT_FILE="/home/sya225/on-policy-distillation/checkpoint_path_${TRAIN_JOB_ID}.txt"

if [ ! -f "$CHECKPOINT_FILE" ] || [ ! -s "$CHECKPOINT_FILE" ]; then
    echo "ERROR: checkpoint path file not found or empty: $CHECKPOINT_FILE" >&2
    exit 1
fi

CHECKPOINT=$(cat "$CHECKPOINT_FILE")

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint directory does not exist: $CHECKPOINT" >&2
    exit 1
fi

echo "Evaluating checkpoint: $CHECKPOINT"
bash eval_math_reasoning.sh "$CHECKPOINT"
