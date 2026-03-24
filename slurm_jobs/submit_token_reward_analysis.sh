#!/bin/bash
#SBATCH -J token-reward-analysis-student0.5b-5000
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=11:59:00
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

echo "Start token reward analysis"

bash on-policy-token-level-reward-analysis.sh

echo "Done"
