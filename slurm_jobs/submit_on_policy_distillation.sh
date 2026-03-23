#!/bin/bash
#SBATCH -J on-policy-teacher14b-student3b-5000-rrg
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --time=2:59:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=rrg-msavva
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/sya225/scratch/on-policy-distillation

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

source test/bin/activate

cd scripts

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
bash rollout.sh &

ROLLOUT_PID=$!

sleep 30

echo "Start training"

CUDA_VISIBLE_DEVICES=1,2,3 \
NPROC_PER_NODE=3 \
bash on-policy-distill.sh \
	Qwen/Qwen2.5-3B-Instruct \
	Qwen/Qwen2.5-14B-Instruct \
	5000

echo "Training finished"

kill $ROLLOUT_PID
wait $ROLLOUT_PID 2>/dev/null
