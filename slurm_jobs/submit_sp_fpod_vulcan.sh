#!/bin/bash -l
#SBATCH -J sp-fpod
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

# SP-FPOD (Student-Prefix Forward-Progressive OPD) — Vulcan, 2 nodes × 4 L40S
#
# Args:
#   $1 = student model (default Qwen/Qwen2.5-0.5B-Instruct)
#   $2 = teacher model (default Qwen/Qwen2.5-14B-Instruct)
#   $3 = data size     (default 8910 — 14B-correctness filtered subset)
#
# vLLM colocate on every GPU; teacher zero3-sharded across 8 ranks.
# Effective batch = 2 × grad_accum_2 × 8 GPU = 32.

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

mkdir -p logs

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-14B-Instruct"}
DATA_SIZE=${3:-8910}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

# Vulcan-specific paths.
export CACHE_BASE=/scratch/sya225/hf_cache
export OUTPUT_BASE_DIR=/scratch/sya225/checkpoints

# Curriculum defaults (callers can override via env at sbatch invocation).
export PREFIX_DIRECTION="${PREFIX_DIRECTION:-forward}"
export PREFIX_K="${PREFIX_K:-5}"
export PREFIX_DELTA="${PREFIX_DELTA:-0.2}"
export PREFIX_P0="${PREFIX_P0:-0.2}"

OUTPUT_DIR="${OUTPUT_BASE_DIR}/sp_fpod_${STUDENT_TAG}_teacher_${TEACHER_TAG}_K${PREFIX_K}_delta${PREFIX_DELTA}_p0${PREFIX_P0}_data${DATA_SIZE}"

# Multi-node distributed rendezvous (torchrun via swift CLI).
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL on Alliance Canada (InfiniBand). Auto-detect typically works.
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# Save the latest checkpoint path on exit (covers SIGTERM/timeout)
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

echo "Start SP-FPOD (2-node colocate): student=$STUDENT_MODEL teacher=$TEACHER_MODEL data=$DATA_SIZE K=$PREFIX_K"
echo "MASTER_ADDR=$MASTER_ADDR  NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE"

# Each node runs one task; swift CLI launches torchrun with NPROC_PER_NODE
# on that node. NODE_RANK is set per task by srun.
srun --output=logs/%N-%j-%t.out bash -c "
    export NODE_RANK=\$SLURM_NODEID
    cd scripts
    bash sp_fpod.sh '$STUDENT_MODEL' '$TEACHER_MODEL' '$DATA_SIZE'
"

echo "Training finished"
