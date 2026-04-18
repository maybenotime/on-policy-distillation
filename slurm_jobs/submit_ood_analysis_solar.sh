#!/bin/bash
#SBATCH -J ood-analysis
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodelist=cs-venus-15
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc
cd /home/sya225/on-policy-distillation
source test_env/bin/activate

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-3B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-7B-Instruct"}
NUM_ROLLOUTS=${3:-10}

cd scripts
bash ood_analysis.sh "$STUDENT_MODEL" "$TEACHER_MODEL" "$NUM_ROLLOUTS"
