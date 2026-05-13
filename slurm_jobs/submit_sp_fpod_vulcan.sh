#!/bin/bash -l
#SBATCH -J sp-fpod
#SBATCH --account=aip-angelx
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

# SP-FPOD (Student-Prefix Forward-Progressive OPD) — Vulcan cluster smoke test
#
# Args:
#   $1 = student model (default Qwen/Qwen2.5-0.5B-Instruct)
#   $2 = teacher model (default Qwen/Qwen2.5-3B-Instruct)
#   $3 = data size     (default 200, override for full runs)
#   $4 = train GPUs    (default 3 — rest of node = GPU 0 for vLLM)
#
# Curriculum env vars (forwarded to sp_fpod.sh):
#   PREFIX_K           number of stages = num_train_epochs (default 5; override to 2 for smoke)
#   PREFIX_DELTA       per-stage delta (default 0.2)
#   PREFIX_P0          forward starting ratio (default 0.2)
#
# Example smoke test:
#   PREFIX_K=2 PREFIX_DELTA=0.4 sbatch slurm_jobs/submit_sp_fpod_vulcan.sh \
#       Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct 200

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

source test/bin/activate

mkdir -p logs

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-3B-Instruct"}
DATA_SIZE=${3:-200}
TRAIN_GPUS=${4:-3}

if [ "$TRAIN_GPUS" -eq 1 ]; then
    TRAIN_CUDA="1"
elif [ "$TRAIN_GPUS" -eq 2 ]; then
    TRAIN_CUDA="1,2"
else
    TRAIN_CUDA="1,2,3"
fi

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

# Vulcan-specific paths (overrides defaults in sp_fpod.sh / rollout.sh).
export CACHE_BASE=/scratch/sya225/hf_cache
export OUTPUT_BASE_DIR=/scratch/sya225/checkpoints

# Curriculum defaults (callers can override via env at sbatch invocation).
export PREFIX_DIRECTION="${PREFIX_DIRECTION:-forward}"
export PREFIX_K="${PREFIX_K:-5}"
export PREFIX_DELTA="${PREFIX_DELTA:-0.2}"
export PREFIX_P0="${PREFIX_P0:-0.2}"

OUTPUT_DIR="${OUTPUT_BASE_DIR}/sp_fpod_${STUDENT_TAG}_teacher_${TEACHER_TAG}_K${PREFIX_K}_delta${PREFIX_DELTA}_p0${PREFIX_P0}_data${DATA_SIZE}"

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

cd scripts

# vLLM rollout server on GPU 0 (background)
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
    bash rollout.sh "$STUDENT_MODEL" &
ROLLOUT_PID=$!

sleep 30

echo "Start SP-FPOD: student=$STUDENT_MODEL teacher=$TEACHER_MODEL data=$DATA_SIZE K=$PREFIX_K"

CUDA_VISIBLE_DEVICES=$TRAIN_CUDA NPROC_PER_NODE=$TRAIN_GPUS \
    bash sp_fpod.sh \
        "$STUDENT_MODEL" \
        "$TEACHER_MODEL" \
        "$DATA_SIZE"

echo "Training finished"

kill $ROLLOUT_PID
wait $ROLLOUT_PID 2>/dev/null
