#!/bin/bash
#SBATCH -J token-reward-analysis-student0.5b-5000
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclusive
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc

cd /home/sya225/on-policy-distillation

source test_env/bin/activate

cd scripts

echo "Start token reward analysis"

bash on-policy-token-level-reward-analysis.sh

echo "Done"
