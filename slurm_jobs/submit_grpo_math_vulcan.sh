#!/bin/bash -l
#SBATCH -J grpo-math
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

# 4-node × 4 L40S colocate GRPO train (16 ranks DDP).
# Reward: built-in `accuracy` ORM (math_verify on \boxed{...} vs solution column).
# Effective batch (per gradient step): 16 ranks × PER_DEVICE_BS × GRAD_ACCUM prompts
#                                   × NUM_GENERATIONS trajectories.
# Default = 16 × 1 × 2 × 8 = 256 trajectories/step.
#
# Args:
#   $1 = model               (default Qwen/Qwen2.5-3B-Instruct)
#   $2 = data size           (default 10000)
#   $3 = num_train_epochs    (default 1)
#
# Env (all optional):
#   PER_DEVICE_BS            default 1
#   GRAD_ACCUM               default 2
#   NUM_GENERATIONS          default 8
#   GRPO_BETA                default 0.04
#   GRPO_TEMPERATURE         default 1.0
#   GRPO_LR                  default 1e-6
#   VLLM_GPU_MEM_UTIL        default 0.4
#   MAX_COMPLETION_LENGTH    default 4096 (set 8192 for 3B+)
#   MAX_LENGTH               default 2×MAX_COMPLETION_LENGTH
#   RESUME_FROM              checkpoint dir to resume from

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source eval_test/bin/activate
mkdir -p logs

MODEL=${1:-Qwen/Qwen2.5-3B-Instruct}
DATA_SIZE=${2:-10000}
NUM_EPOCHS=${3:-1}

export CACHE_BASE=/scratch/sya225/hf_cache
export OUTPUT_BASE_DIR=/scratch/sya225/checkpoints
export HF_HOME=$CACHE_BASE
# Per-job /tmp dirs avoid /scratch NFS FileLock race (Errno 116).
export MODELSCOPE_CACHE=/tmp/sya225_grpo_ms_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/tmp/sya225_grpo_ds_$SLURM_JOB_ID
export TMPDIR=/tmp/sya225_grpo_tmp_$SLURM_JOB_ID
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket
export NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$MODELSCOPE_CACHE/lockers" "$HF_DATASETS_CACHE" "$TMPDIR"

export PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export GRPO_BETA="${GRPO_BETA:-0.04}"
export GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"
export GRPO_LR="${GRPO_LR:-1e-6}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.4}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
export MAX_LENGTH="${MAX_LENGTH:-$((MAX_COMPLETION_LENGTH * 2))}"
export RESUME_FROM="${RESUME_FROM:-}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29557
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_TAG=$(basename "$MODEL")
OUTPUT_DIR="${OUTPUT_BASE_DIR}/grpo_${MODEL_TAG}_epochs${NUM_EPOCHS}_data${DATA_SIZE}"

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

echo "[grpo-math] job=$SLURM_JOB_ID nodes=$NNODES"
echo "  model=$MODEL data=$DATA_SIZE epochs=$NUM_EPOCHS"
echo "  pdbs=$PER_DEVICE_BS ga=$GRAD_ACCUM num_gen=$NUM_GENERATIONS"
echo "  beta=$GRPO_BETA T=$GRPO_TEMPERATURE lr=$GRPO_LR"
echo "  eff_prompts/step=$((16*PER_DEVICE_BS*GRAD_ACCUM)) eff_traj/step=$((16*PER_DEVICE_BS*GRAD_ACCUM*NUM_GENERATIONS))"
echo "  resume_from=${RESUME_FROM:-<fresh>}"
echo "  master_addr=$MASTER_ADDR"
echo "  output_dir=$OUTPUT_DIR"

srun --output=logs/%N-%j-%t.out bash -c "
    export NODE_RANK=\$SLURM_NODEID
    mkdir -p \"$MODELSCOPE_CACHE/lockers\" \"$HF_DATASETS_CACHE\" \"$TMPDIR\"
    cd scripts
    bash grpo_math.sh '$MODEL' '$DATA_SIZE' '$NUM_EPOCHS'
"

echo "Training finished"
