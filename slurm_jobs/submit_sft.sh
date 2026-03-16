#!/bin/bash
#SBATCH -J sft-student0.5b
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=def-angelx

cd /home/sya225/scratch/representation-align

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

source test/bin/activate


echo "Start training"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash SFT.sh
