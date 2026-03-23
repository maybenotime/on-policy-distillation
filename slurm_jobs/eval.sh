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

CHECKPOINT=$(cat /home/sya225/scratch/on-policy-distillation/checkpoint_path_${TRAIN_JOB_ID}.txt)
echo "Evaluating checkpoint: $CHECKPOINT"
bash eval_math_reasoning.sh "$CHECKPOINT"
