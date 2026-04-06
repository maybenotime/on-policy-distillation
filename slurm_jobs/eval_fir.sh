#!/bin/bash
#SBATCH -J on-policy-teacher7b-student3b-5000-eval
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:59:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=288G
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=rrg-msavva
#SBATCH --mail-user=yangshipingnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/sya225/scratch/on-policy-distillation

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

source test/bin/activate

cd scripts

CHECKPOINT_FILE="/home/sya225/scratch/on-policy-distillation/checkpoint_path_${TRAIN_JOB_ID}.txt"

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
