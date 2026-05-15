#!/bin/bash -l
#SBATCH -J opd-base
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Plain GKD baseline (no prefix curriculum) — Vulcan, 2 nodes × 4 L40S
#
# Args:
#   $1 = student model    (default Qwen/Qwen2.5-0.5B-Instruct)
#   $2 = teacher model    (default Qwen/Qwen2.5-14B-Instruct)
#   $3 = data size        (default 8910 — matches SP/TP)
#   $4 = num_train_epochs (default 1; pass 5 for matched-epoch baseline)
#
# Effective batch = 2 × grad_accum_2 × 8 GPU = 32.

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

mkdir -p logs

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-14B-Instruct"}
DATA_SIZE=${3:-8910}
NUM_EPOCHS=${4:-1}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

# Vulcan-specific paths.
export CACHE_BASE=/scratch/sya225/hf_cache
export OUTPUT_BASE_DIR=/scratch/sya225/checkpoints

OUTPUT_DIR="${OUTPUT_BASE_DIR}/opd_baseline_${STUDENT_TAG}_teacher_${TEACHER_TAG}_epochs${NUM_EPOCHS}_data${DATA_SIZE}"

# Multi-node distributed rendezvous.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29502
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export HF_DATASETS_CACHE=/tmp/sya225_ds_cache_$SLURM_JOB_ID
mkdir -p "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${REPO_DIR}/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint saved by trap: $CHECKPOINT"
    else
        echo "No checkpoint found at exit" >&2
    fi
}
trap save_checkpoint EXIT

echo "Start OPD baseline (2-node colocate, $NUM_EPOCHS epochs):"
echo "  student=$STUDENT_MODEL teacher=$TEACHER_MODEL data=$DATA_SIZE"
echo "  MASTER_ADDR=$MASTER_ADDR  NNODES=$NNODES"

srun --output=logs/%N-%j-%t.out bash -c "
    export NODE_RANK=\$SLURM_NODEID
    cd scripts
    bash opd_baseline.sh '$STUDENT_MODEL' '$TEACHER_MODEL' '$DATA_SIZE' '$NUM_EPOCHS'
"

echo "Training finished"
