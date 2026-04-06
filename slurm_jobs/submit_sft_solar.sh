#!/bin/bash
#SBATCH -J sft-student0.5b-80000
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc

cd /home/sya225/on-policy-distillation

source test_env/bin/activate

cd scripts


echo "Start training"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash SFT.sh Qwen/Qwen2.5-0.5B-Instruct
