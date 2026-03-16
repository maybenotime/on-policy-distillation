#!/bin/bash
#SBATCH -J on-policy-teacher7b-student0.5b-eval
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=288G
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=def-angelx

cd /home/sya225/scratch/representation-align

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

source test/bin/activate

bash eval_math_reasoning.sh /home/sya225/scratch/representation-align/output/v1-20260306-115644/checkpoint-417
