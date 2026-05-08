#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=exp37-opsd-3b-14b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# OPSD Ablation — 3B student, 14B teacher context (Plan 37, Exp C)
# Mirrors exp36 3B←14B for direct comparison.
# Note: 3B full fine-tuning + colocate vLLM on 1×H100 (80GB).
# vllm_gpu_memory_utilization=0.35 to leave more room for training.

set -euo pipefail

BASE=/home/cta156/scratch/on-policy-distillation

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export MODEL=Qwen/Qwen2.5-3B-Instruct
export OPSD_MODE=single
export OPSD_TEACHER_MODEL=Qwen2.5-14B-Instruct
export OPSD_DATA_PATH="${BASE}/data/math_all_teachers_correct.jsonl"
export DATA_SIZE=840

echo "Student:         $MODEL"
echo "Teacher context: $OPSD_TEACHER_MODEL"
echo "Data size:       $DATA_SIZE"

save_checkpoint() {
    OUTPUT_BASE="${BASE}/output/opsd_Qwen2.5-3B-Instruct_modesingle_teacherQwen2.5-14B-Instruct"
    CHECKPOINT=$(find "$OUTPUT_BASE" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint: $CHECKPOINT"
    else
        echo "No checkpoint found" >&2
    fi
}
trap save_checkpoint EXIT

cd "${BASE}/scripts"
# Lower vLLM utilization for 3B to avoid OOM with colocate mode
VLLM_UTIL=0.35 bash opsd.sh

echo "OPSD training done."
