#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=exp37-opsd-0.5b-14b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# OPSD Ablation — 0.5B student, 14B teacher context (Plan 37, Exp A)
# Controls for recovery failure: teacher sees correct answer → no OOD problem.
# Mirrors exp36 0.5B←14B for direct comparison (S√T√ vs OPSD vs standard GKD).
#
# OPSD mechanics: student=teacher=0.5B, but teacher copy gets correct 14B answer
# as privileged context. 840 samples, 200 steps, colocate vLLM (1×H100).

set -euo pipefail

BASE=/home/cta156/scratch/on-policy-distillation

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export MODEL=Qwen/Qwen2.5-0.5B-Instruct
export OPSD_MODE=single
export OPSD_TEACHER_MODEL=Qwen2.5-14B-Instruct
export OPSD_DATA_PATH="${BASE}/data/math_all_teachers_correct.jsonl"
export DATA_SIZE=840

echo "Student:         $MODEL"
echo "Teacher context: $OPSD_TEACHER_MODEL"
echo "Data size:       $DATA_SIZE"

save_checkpoint() {
    OUTPUT_BASE="${BASE}/output/opsd_Qwen2.5-0.5B-Instruct_modesingle_teacherQwen2.5-14B-Instruct"
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
bash opsd.sh

echo "OPSD training done."
