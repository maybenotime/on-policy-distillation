#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=at-data-gen-dryrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Data generation for AT experiments (Dryrun - 8 samples)
cd /home/cta156/scratch/on-policy-distillation
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

# Use hf_transfer for fast loading if needed
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Starting AT hybrid data generation (dryrun)..."

python source/generate_hybrid_trajectories.py \
    --input data/prompts_math_swift.jsonl \
    --output data/hybrid_at_30_dryrun.jsonl \
    --student Qwen/Qwen2.5-0.5B-Instruct \
    --teacher Qwen/Qwen2.5-14B-Instruct \
    --ratio 0.3 \
    --limit 8

echo "Dryrun generation finished."
