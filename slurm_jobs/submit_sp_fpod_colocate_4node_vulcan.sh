#!/bin/bash -l
#SBATCH -J sp-fpod-4n
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

# 4-node × 4 L40S colocate prefix_curriculum train (16 ranks DDP).
# Effective batch = 16 × PER_DEVICE_BS × GRAD_ACCUM (default 16×1×2=32).
#
# Args: $1 = student model, $2 = teacher model, $3 = data size
#
# Env vars (all optional, with defaults):
#   PREFIX_DIRECTION   forward | backward | entropy_topk | entropy_reweight
#   PREFIX_K           number of stages, default 1
#   PREFIX_DELTA       per-stage delta, default 0
#   PREFIX_P0          forward starting ratio, default 0.2
#   PREFIX_TOPK_FRAC   entropy_topk fraction, default 0.2
#   PREFIX_REWEIGHT_TOPK  entropy_reweight top-K, default 16
#   PER_DEVICE_BS      default 1 (16 ranks × 1 × 2 = 32)
#   GRAD_ACCUM         default 2
#   RESUME_FROM        ckpt path to resume from (default unset)

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source eval_test/bin/activate
mkdir -p logs

STUDENT_MODEL=${1:?student required}
TEACHER_MODEL=${2:?teacher required}
DATA_SIZE=${3:?data size required}

export CACHE_BASE=/scratch/sya225/hf_cache
export OUTPUT_BASE_DIR=/scratch/sya225/checkpoints
export HF_HOME=$CACHE_BASE
export MODELSCOPE_CACHE=/tmp/sya225_spfpod4n_ms_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/tmp/sya225_spfpod4n_ds_$SLURM_JOB_ID
export TMPDIR=/tmp/sya225_spfpod4n_tmp_$SLURM_JOB_ID
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket
export NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$MODELSCOPE_CACHE/lockers" "$HF_DATASETS_CACHE" "$TMPDIR"

export PREFIX_DIRECTION="${PREFIX_DIRECTION:-forward}"
export PREFIX_K="${PREFIX_K:-1}"
export PREFIX_DELTA="${PREFIX_DELTA:-0}"
export PREFIX_P0="${PREFIX_P0:-0.2}"
export PREFIX_TOPK_FRAC="${PREFIX_TOPK_FRAC:-0.2}"
export PREFIX_REWEIGHT_TOPK="${PREFIX_REWEIGHT_TOPK:-16}"
export PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export RESUME_FROM="${RESUME_FROM:-}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29556
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")
OUTPUT_DIR="${OUTPUT_BASE_DIR}/sp_fpod_${STUDENT_TAG}_teacher_${TEACHER_TAG}_K${PREFIX_K}_delta${PREFIX_DELTA}_p0${PREFIX_P0}_data${DATA_SIZE}"

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

echo "[sp-fpod-4n] job=$SLURM_JOB_ID nodes=$NNODES"
echo "  direction=$PREFIX_DIRECTION K=$PREFIX_K delta=$PREFIX_DELTA p0=$PREFIX_P0 topk_frac=$PREFIX_TOPK_FRAC reweight_K=$PREFIX_REWEIGHT_TOPK"
echo "  student=$STUDENT_MODEL teacher=$TEACHER_MODEL data=$DATA_SIZE"
echo "  pdbs=$PER_DEVICE_BS ga=$GRAD_ACCUM eff=$((16*PER_DEVICE_BS*GRAD_ACCUM))"
echo "  resume_from=${RESUME_FROM:-<fresh>}"
echo "  master_addr=$MASTER_ADDR"
echo "  output_dir=$OUTPUT_DIR"

srun --output=logs/%N-%j-%t.out bash -c "
    export NODE_RANK=\$SLURM_NODEID
    mkdir -p \"$MODELSCOPE_CACHE/lockers\" \"$HF_DATASETS_CACHE\" \"$TMPDIR\"
    cd scripts
    bash sp_fpod.sh '$STUDENT_MODEL' '$TEACHER_MODEL' '$DATA_SIZE'
"

echo "Training finished"
