#!/bin/bash -l
#SBATCH -J opd-vul-col
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Vulcan 1-node × 4 L40S colocate OPD wrapper.
# Effective batch = 4 × PER_DEVICE_BS × GRAD_ACCUM (default 4×2×4=32).
#
# Args: $1 student, $2 teacher, $3 data_size, $4 num_epochs
#
# Env (optional):
#   OPD_DATASET           full dataset spec
#   OUTPUT_TAG_SUFFIX     default data${DATA_SIZE}
#   PER_DEVICE_BS         default 2
#   GRAD_ACCUM            default 4
#   TEACHER_DEEPSPEED     default unset (set "zero3" for 14B teacher on L40S 48GB)
#   VLLM_GPU_MEM_UTIL     default 0.3
#   MAX_COMPLETION_LENGTH default 4096

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
export MODELSCOPE_CACHE=/tmp/sya225_vcol_ms_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/tmp/sya225_vcol_ds_$SLURM_JOB_ID
mkdir -p "$MODELSCOPE_CACHE/lockers" "$HF_DATASETS_CACHE"

OUTPUT_TAG_SUFFIX="${OUTPUT_TAG_SUFFIX:-data${DATA_SIZE}}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/opd_vulcol_${STUDENT_TAG}_teacher_${TEACHER_TAG}_epochs${NUM_EPOCHS}_${OUTPUT_TAG_SUFFIX}"

export PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export TEACHER_DEEPSPEED="${TEACHER_DEEPSPEED:-}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.3}"
export OPD_DATASET="${OPD_DATASET:-${REPO_DIR}/data/prompts_math_swift.jsonl#${DATA_SIZE}}"
export OUTPUT_BASE_DIR
export OUTPUT_TAG_SUFFIX

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

echo "[opd-vulcol] job=$SLURM_JOB_ID"
echo "  student=$STUDENT_MODEL teacher=$TEACHER_MODEL data=$DATA_SIZE"
echo "  pdbs=$PER_DEVICE_BS ga=$GRAD_ACCUM eff=$((4*PER_DEVICE_BS*GRAD_ACCUM))"
echo "  teacher_ds=${TEACHER_DEEPSPEED:-<unset>} vllm_mem=$VLLM_GPU_MEM_UTIL"
echo "  output=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
    bash scripts/opd_baseline.sh "$STUDENT_MODEL" "$TEACHER_MODEL" "$DATA_SIZE" "$NUM_EPOCHS"
