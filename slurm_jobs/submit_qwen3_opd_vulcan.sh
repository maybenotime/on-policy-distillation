#!/bin/bash
#SBATCH -J qwen3-opd
#SBATCH --account=sya225
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:00:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Qwen3 on-policy distillation (non-thinking mode) — Vulcan cluster
#
# Args:
#   $1 = student model (default Qwen/Qwen3-1.7B)
#   $2 = teacher model (default Qwen/Qwen3-8B)
#   $3 = dataset path (default data/prompts_math_swift.jsonl)
#   $4 = data size  (default 5000)
#   $5 = train GPUs (default 3)

REPO_DIR=/project/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

source test/bin/activate

mkdir -p logs

STUDENT_MODEL=${1:-"Qwen/Qwen3-1.7B"}
TEACHER_MODEL=${2:-"Qwen/Qwen3-8B"}
DATASET=${3:-"data/prompts_math_swift.jsonl"}
DATA_SIZE=${4:-5000}
TRAIN_GPUS=${5:-3}

if [ "$TRAIN_GPUS" -eq 1 ]; then
    TRAIN_CUDA="1"
elif [ "$TRAIN_GPUS" -eq 2 ]; then
    TRAIN_CUDA="1,2"
else
    TRAIN_CUDA="1,2,3"
fi

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

OUTPUT_BASE="/scratch/sya225/output/qwen3_student_${STUDENT_TAG}_teacher_${TEACHER_TAG}_data${DATA_SIZE}"

# Save the latest checkpoint path on exit (covers SIGTERM/timeout)
save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT_BASE" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${REPO_DIR}/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint saved by trap: $CHECKPOINT"
    else
        echo "No checkpoint found at exit" >&2
    fi
}
trap save_checkpoint EXIT

cd scripts

# Cache lives on /scratch (NOT /home — Vulcan home quota is small).
export CACHE_BASE=/scratch/sya225/cache
export OUTPUT_BASE_DIR=/scratch/sya225/output

# vLLM rollout server on GPU 0 (background)
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
    bash rollout_qwen3.sh "$STUDENT_MODEL" &
ROLLOUT_PID=$!

sleep 30

echo "Start training: dataset=$DATASET, size=$DATA_SIZE, student=$STUDENT_MODEL, teacher=$TEACHER_MODEL"

CUDA_VISIBLE_DEVICES=$TRAIN_CUDA NPROC_PER_NODE=$TRAIN_GPUS \
    bash on-policy-distill_qwen3.sh \
        "$STUDENT_MODEL" \
        "$TEACHER_MODEL" \
        "$DATA_SIZE" \
        "../${DATASET}"

echo "Training finished"

kill $ROLLOUT_PID
wait $ROLLOUT_PID 2>/dev/null
