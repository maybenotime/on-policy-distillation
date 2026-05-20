#!/bin/bash -l
#SBATCH -J opsd
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Vulcan 2-node × 4 L40S colocate OPSD. Eff batch = 8 × PER_DEVICE_BS × GRAD_ACCUM
# (default 8 × 2 × 2 = 32).
#
# Args:
#   $1 = student/teacher model   (default Qwen/Qwen2.5-0.5B-Instruct)
#   $2 = data size               (default 7000)
#   $3 = num_train_epochs        (default 1)
#
# Env:
#   OPSD_MODE                    default single
#   OPSD_TEACHER_MODEL           default Qwen2.5-14B-Instruct  (override per-job!)
#   OPSD_DATA_PATH               default /scratch/sya225/data/fmd_q2both_correct_7k.jsonl
#   PER_DEVICE_BS                default 2
#   GRAD_ACCUM                   default 2
#   MAX_LENGTH                   default 8192
#   MAX_COMPLETION_LENGTH        default 4096
#   OPSD_LR                      default 2e-5
#   VLLM_GPU_MEM_UTIL            default 0.4

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source eval_test/bin/activate
mkdir -p logs

STUDENT_MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
DATA_SIZE=${2:-7000}
NUM_EPOCHS=${3:-1}
STUDENT_TAG=$(basename "$STUDENT_MODEL")

export CACHE_BASE=/scratch/sya225/hf_cache
export OUTPUT_BASE_DIR=/scratch/sya225/checkpoints
export HF_HOME=$CACHE_BASE
# Per-job /tmp dirs avoid /scratch NFS FileLock race (Errno 116).
export MODELSCOPE_CACHE=/tmp/sya225_opsd_ms_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/tmp/sya225_opsd_ds_$SLURM_JOB_ID
export TMPDIR=/tmp/sya225_opsd_tmp_$SLURM_JOB_ID
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$MODELSCOPE_CACHE/lockers" "$HF_DATASETS_CACHE" "$TMPDIR"

# OPSD-specific knobs
export OPSD_MODE="${OPSD_MODE:-single}"
export OPSD_TEACHER_MODEL="${OPSD_TEACHER_MODEL:-Qwen2.5-14B-Instruct}"
export OPSD_DATA_PATH="${OPSD_DATA_PATH:-/scratch/sya225/data/fmd_q2both_correct_7k.jsonl}"

# Training knobs (defaults sized for 2-node × 4 GPU = 8 ranks, eff batch 32)
export PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export MAX_LENGTH="${MAX_LENGTH:-8192}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
export OPSD_LR="${OPSD_LR:-2e-5}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.4}"
export EPOCHS_MODE=1   # use --num_train_epochs (not --max_steps)

# Tag suffix so output dir distinguishes 7B vs 14B teacher runs and student size.
export OUTPUT_TAG_SUFFIX="${OUTPUT_TAG_SUFFIX:-q2both${DATA_SIZE}}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29558
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# OPSD output dir pattern (mirrors scripts/opsd.sh)
if [ "$OPSD_MODE" = "multi" ]; then
    TEACHER_TAG="multi_$(echo "$OPSD_TEACHER_MODELS" | tr ',' '_' | tr -d ' ')"
else
    TEACHER_TAG="$OPSD_TEACHER_MODEL"
fi
OUTPUT_DIR="${OUTPUT_BASE_DIR}/opsd_${STUDENT_TAG}_mode${OPSD_MODE}_teacher${TEACHER_TAG}_${OUTPUT_TAG_SUFFIX}"

save_checkpoint() {
    CKPT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CKPT" ]; then
        echo "$CKPT" > "${REPO_DIR}/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint saved by trap: $CKPT"
    else
        echo "No checkpoint found at exit" >&2
    fi
}
trap save_checkpoint EXIT

echo "[opsd] job=$SLURM_JOB_ID nodes=$NNODES"
echo "  student/teacher=$STUDENT_MODEL"
echo "  OPSD_TEACHER_MODEL=$OPSD_TEACHER_MODEL  (rationale source)"
echo "  OPSD_DATA_PATH=$OPSD_DATA_PATH  data=$DATA_SIZE epochs=$NUM_EPOCHS"
echo "  pdbs=$PER_DEVICE_BS ga=$GRAD_ACCUM eff=$((8 * PER_DEVICE_BS * GRAD_ACCUM))"
echo "  max_len=$MAX_LENGTH max_completion=$MAX_COMPLETION_LENGTH lr=$OPSD_LR vllm_mem=$VLLM_GPU_MEM_UTIL"
echo "  output_dir=$OUTPUT_DIR"

srun --output=logs/%N-%j-%t.out bash -c "
    export NODE_RANK=\$SLURM_NODEID
    mkdir -p \"$MODELSCOPE_CACHE/lockers\" \"$HF_DATASETS_CACHE\" \"$TMPDIR\"
    cd scripts
    bash opsd.sh '$STUDENT_MODEL' '$DATA_SIZE' '$NUM_EPOCHS'
"

echo "Training finished"
