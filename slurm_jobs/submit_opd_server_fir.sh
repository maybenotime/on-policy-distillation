#!/bin/bash -l
#SBATCH -J opd-q2-srv
#SBATCH --account=rrg-angelx
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=192
#SBATCH --time=11:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Q2.5 OPD with SEPARATE vLLM rollout server (server mode), Fir 4×H100 80GB.
#
# Layout:
#   GPU 0: vllm rollout server (always-on, no sleep/wakeup overhead)
#   GPU 1-3: training (3-rank DDP)
#
# Teacher Q2.5-32B BF16 fits in 80GB H100 directly (no zero3 needed).
# Effective batch = 3 ranks × per_device_bs × grad_accum (target ~32).
#
# Args:
#   $1 = student model    (default Qwen/Qwen2.5-0.5B-Instruct)
#   $2 = teacher model    (default Qwen/Qwen2.5-32B-Instruct)
#   $3 = data size        (default 10000)
#   $4 = num_train_epochs (default 1)

REPO_DIR=/home/sya225/scratch/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

mkdir -p logs

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-32B-Instruct"}
DATA_SIZE=${3:-10000}
NUM_EPOCHS=${4:-1}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

export CACHE_BASE=/home/sya225/scratch/hf_cache
export OUTPUT_BASE_DIR=/home/sya225/scratch/checkpoints
export HF_DATASETS_CACHE=/tmp/sya225_ds_cache_$SLURM_JOB_ID
export MODELSCOPE_CACHE=/tmp/sya225_ms_cache_$SLURM_JOB_ID
mkdir -p "$HF_DATASETS_CACHE" "$MODELSCOPE_CACHE" "$MODELSCOPE_CACHE/lockers"

OUTPUT_TAG_SUFFIX="${OUTPUT_TAG_SUFFIX:-data${DATA_SIZE}}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/opd_baseline_${STUDENT_TAG}_teacher_${TEACHER_TAG}_epochs${NUM_EPOCHS}_${OUTPUT_TAG_SUFFIX}"

VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-3}"

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

echo "Start OPD Q2.5 SERVER-mode on Fir (single-node 4×H100, $NUM_EPOCHS epochs)"
echo "  student=$STUDENT_MODEL teacher=$TEACHER_MODEL data=$DATA_SIZE"
echo "  per_device_bs=$PER_DEVICE_BS grad_accum=$GRAD_ACCUM eff_batch=$((3*PER_DEVICE_BS*GRAD_ACCUM))"
echo "  vLLM rollout port=$VLLM_SERVER_PORT"

# Start rollout server on GPU 0 in background
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
    bash scripts/rollout.sh "$STUDENT_MODEL" > logs/rollout-${SLURM_JOB_ID}.log 2>&1 &
ROLLOUT_PID=$!
echo "[rollout] PID=$ROLLOUT_PID, log=logs/rollout-${SLURM_JOB_ID}.log"

# Wait for server ready (poll log for uvicorn startup line, no /v1/models endpoint)
echo "[rollout] waiting for server up..."
for i in $(seq 1 60); do
    if grep -q "Application startup complete" "logs/rollout-${SLURM_JOB_ID}.log" 2>/dev/null; then
        echo "[rollout] up after ${i}0s"
        break
    fi
    sleep 10
done
if ! grep -q "Application startup complete" "logs/rollout-${SLURM_JOB_ID}.log" 2>/dev/null; then
    echo "[rollout] FAILED to start after 10min; aborting"
    kill $ROLLOUT_PID 2>/dev/null
    exit 1
fi
if ! kill -0 $ROLLOUT_PID 2>/dev/null; then
    echo "[rollout] PID dead; aborting"
    exit 1
fi

# Train on GPUs 1-3 (3 ranks)
export PER_DEVICE_BS GRAD_ACCUM VLLM_SERVER_PORT
CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 \
    bash scripts/opd_baseline_server.sh "$STUDENT_MODEL" "$TEACHER_MODEL" "$DATA_SIZE" "$NUM_EPOCHS"
TRAIN_EXIT=$?

echo "Training exit=$TRAIN_EXIT; shutting down rollout server"
kill $ROLLOUT_PID 2>/dev/null
wait $ROLLOUT_PID 2>/dev/null

exit $TRAIN_EXIT
