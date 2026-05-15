#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=exp37-opsd-dryrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# OPSD Ablation dry run — 0.5B student, 14B teacher context, 5 steps only
# Verifies: opsd_plugin.py loads, colocate vLLM starts, loss is non-zero

set -euo pipefail

BASE=/home/cta156/scratch/on-policy-distillation

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export MODEL=Qwen/Qwen2.5-0.5B-Instruct
export OPSD_MODE=single
export OPSD_TEACHER_MODEL=Qwen2.5-14B-Instruct
export OPSD_DATA_PATH="${BASE}/data/math_all_teachers_correct.jsonl"
export DATA_SIZE=20

echo "Student:         $MODEL"
echo "Teacher context: $OPSD_TEACHER_MODEL"
echo "Data size:       $DATA_SIZE"
echo "Mode:            $OPSD_MODE"

cd "${BASE}/scripts"
MAX_STEPS=5 bash opsd.sh

echo "Dry run complete."
