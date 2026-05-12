#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=at-data-gen-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Full data generation for AT experiments (30%, 50%, 70%)
cd /home/cta156/scratch/on-policy-distillation
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Starting AT hybrid data generation (Full 5000 samples)..."

# Generate 30% Truncation
python source/generate_hybrid_trajectories.py \
    --input data/prompts_math_swift.jsonl \
    --output data/hybrid_at_30.jsonl \
    --student Qwen/Qwen2.5-0.5B-Instruct \
    --teacher Qwen/Qwen2.5-14B-Instruct \
    --ratio 0.3

# Generate 50% Truncation
python source/generate_hybrid_trajectories.py \
    --input data/prompts_math_swift.jsonl \
    --output data/hybrid_at_50.jsonl \
    --student Qwen/Qwen2.5-0.5B-Instruct \
    --teacher Qwen/Qwen2.5-14B-Instruct \
    --ratio 0.5

# Generate 70% Truncation
python source/generate_hybrid_trajectories.py \
    --input data/prompts_math_swift.jsonl \
    --output data/hybrid_at_70.jsonl \
    --student Qwen/Qwen2.5-0.5B-Instruct \
    --teacher Qwen/Qwen2.5-14B-Instruct \
    --ratio 0.7

echo "All hybrid data generation finished."
