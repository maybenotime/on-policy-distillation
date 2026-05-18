#!/bin/bash -l
#SBATCH -J opd-4n-col
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Vulcan 4-node × 4 L40S colocate OPD. 16 train ranks DDP.
# Effective batch = 16 × PER_DEVICE_BS × GRAD_ACCUM (default 16×1×2=32).

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source eval_test/bin/activate
mkdir -p logs

STUDENT_MODEL=${1:?student required}
TEACHER_MODEL=${2:?teacher required}
DATA_SIZE=${3:?data size required}
NUM_EPOCHS=${4:?num epochs required}
STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

export CACHE_BASE=/scratch/sya225/hf_cache
export OUTPUT_BASE_DIR=/scratch/sya225/checkpoints
export HF_HOME=$CACHE_BASE
export MODELSCOPE_CACHE=/tmp/sya225_4ncol_ms_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/tmp/sya225_4ncol_ds_$SLURM_JOB_ID
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket
export NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$MODELSCOPE_CACHE/lockers" "$HF_DATASETS_CACHE"

OUTPUT_TAG_SUFFIX="${OUTPUT_TAG_SUFFIX:-data${DATA_SIZE}}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/opd_4ncol_${STUDENT_TAG}_teacher_${TEACHER_TAG}_epochs${NUM_EPOCHS}_${OUTPUT_TAG_SUFFIX}"

export PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export TEACHER_DEEPSPEED="${TEACHER_DEEPSPEED:-}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.3}"
export OPD_DATASET="${OPD_DATASET:-${REPO_DIR}/data/prompts_math_swift.jsonl#${DATA_SIZE}}"
export OUTPUT_BASE_DIR
export OUTPUT_TAG_SUFFIX

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29555
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

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

echo "[opd-4ncol] job=$SLURM_JOB_ID nodes=$NNODES"
echo "  student=$STUDENT_MODEL teacher=$TEACHER_MODEL data=$DATA_SIZE"
echo "  pdbs=$PER_DEVICE_BS ga=$GRAD_ACCUM eff=$((16*PER_DEVICE_BS*GRAD_ACCUM))"
echo "  teacher_ds=${TEACHER_DEEPSPEED:-<unset>} vllm_mem=$VLLM_GPU_MEM_UTIL"
echo "  master_addr=$MASTER_ADDR"

srun --output=logs/%N-%j-%t.out bash -c "
    export NODE_RANK=\$SLURM_NODEID
    mkdir -p \"$MODELSCOPE_CACHE/lockers\" \"$HF_DATASETS_CACHE\"
    cd scripts
    bash opd_baseline.sh '$STUDENT_MODEL' '$TEACHER_MODEL' '$DATA_SIZE' '$NUM_EPOCHS'
"

echo "Training finished"
