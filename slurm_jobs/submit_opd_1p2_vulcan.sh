#!/bin/bash -l
#SBATCH -J opd-1p2
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:59:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=3
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/%N-%j.out

# OPD 1+2 server-mode training wrapper for Vulcan (verified 2026-05-18).
#
# Layout (3 nodes x 4 L40S):
#   Node 0 (NODES[0]): vllm rollout server only (4 GPUs visible, TP via ROLLOUT_TP)
#   Node 1-2 (NODES[1], NODES[2]): train DDP, 8 ranks total (4/node), student+teacher
#
# ~30s/step (3x faster than 2-node colocate).
#
# Args:
#   $1 = student model    (e.g. Qwen/Qwen2.5-3B-Instruct)
#   $2 = teacher model    (e.g. Qwen/Qwen2.5-14B-Instruct)
#   $3 = data size        (used in --dataset path + tag, e.g. 7000)
#   $4 = num_train_epochs (e.g. 1)
#
# Env vars (all optional):
#   OPD_DATASET           full dataset spec, default data/prompts_math_swift.jsonl#${DATA_SIZE}
#   OUTPUT_TAG_SUFFIX     default data${DATA_SIZE}
#   ROLLOUT_TP            vllm_tensor_parallel_size for rollout, default 1
#                         (Q2.5-0.5B has 14 heads -> only TP=1; Q2.5-3B/Q3-* OK with TP=4)
#   TEACHER_DEEPSPEED     pass-through to swift rlhf --teacher_deepspeed, default unset
#                         (use zero3 for 14B teacher on L40S 48GB)
#   MAX_COMPLETION_LENGTH default 4096
#   ENABLE_THINKING       pass through (false for Q3), default unset (Q2.5)
#   PER_DEVICE_BS         default 2
#   GRAD_ACCUM            default 2

set -u

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source eval_test/bin/activate

mkdir -p logs

STUDENT_MODEL=${1:?student model required}
TEACHER_MODEL=${2:?teacher model required}
DATA_SIZE=${3:?data size required}
NUM_EPOCHS=${4:?num epochs required}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

OPD_DATASET="${OPD_DATASET:-${REPO_DIR}/data/prompts_math_swift.jsonl#${DATA_SIZE}}"
OUTPUT_TAG_SUFFIX="${OUTPUT_TAG_SUFFIX:-data${DATA_SIZE}}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
TEACHER_DEEPSPEED="${TEACHER_DEEPSPEED:-}"

OUTPUT_DIR="/scratch/sya225/checkpoints/opd_1p2_${STUDENT_TAG}_teacher_${TEACHER_TAG}_epochs${NUM_EPOCHS}_${OUTPUT_TAG_SUFFIX}"

NODES=($(scontrol show hostnames $(squeue -j $SLURM_JOB_ID -h -o '%N')))
ROLLOUT_NODE=${NODES[0]}
TRAIN_A=${NODES[1]}
TRAIN_B=${NODES[2]}

export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket
export NCCL_ASYNC_ERROR_HANDLING=1
export MODELSCOPE_CACHE=/tmp/sya225_1p2_ms_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/tmp/sya225_1p2_ds_$SLURM_JOB_ID
export TMPDIR=/tmp/sya225_1p2_tmp_$SLURM_JOB_ID
export HF_HOME=/scratch/sya225/hf_cache

LOG_DIR=/scratch/sya225/tmp/1p2_${SLURM_JOB_ID}
mkdir -p "$LOG_DIR"

echo "[opd-1p2] job=$SLURM_JOB_ID"
echo "  rollout=$ROLLOUT_NODE  train=$TRAIN_A,$TRAIN_B"
echo "  student=$STUDENT_MODEL  teacher=$TEACHER_MODEL"
echo "  dataset=$OPD_DATASET  epochs=$NUM_EPOCHS"
echo "  rollout_tp=$ROLLOUT_TP  teacher_ds=${TEACHER_DEEPSPEED:-<unset>}  thinking=${ENABLE_THINKING:-<unset>}"
echo "  output_dir=$OUTPUT_DIR"

save_checkpoint() {
    CKPT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CKPT" ]; then
        echo "$CKPT" > "${REPO_DIR}/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "[trap] checkpoint saved: $CKPT"
    else
        echo "[trap] no checkpoint at exit" >&2
    fi
}
trap save_checkpoint EXIT

# -------- Launch rollout on NODES[0] (single task, all 4 GPUs) --------
srun --jobid=$SLURM_JOB_ID --overlap -w "$ROLLOUT_NODE" -N1 -n1 --gres=gpu:l40s:4 \
    --output=$LOG_DIR/rollout.log bash -c "
set -u
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
cd $REPO_DIR
source eval_test/bin/activate
export NCCL_IB_DISABLE=1 NCCL_NET=Socket
mkdir -p $MODELSCOPE_CACHE/lockers
export MODELSCOPE_CACHE=$MODELSCOPE_CACHE
export HF_HOME=$HF_HOME
if [ -n \"${ENABLE_THINKING:-}\" ]; then THINK_FLAG=\"--enable_thinking ${ENABLE_THINKING}\"; else THINK_FLAG=\"\"; fi
exec swift rollout \
    --model $STUDENT_MODEL \
    --vllm_tensor_parallel_size $ROLLOUT_TP \
    --use_hf true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --padding_free true \
    --host 0.0.0.0 --port 8000 \
    --load_from_cache_file false \$THINK_FLAG
" &
ROLLOUT_PID=$!
echo "[rollout] launched, PID=$ROLLOUT_PID"

# -------- Wait for rollout ready (up to 30 min) --------
echo "[rollout] waiting for 'Application startup complete'..."
UP=0
for i in $(seq 1 180); do
    if grep -q "Application startup complete" "$LOG_DIR/rollout.log" 2>/dev/null; then
        echo "[rollout] up after ${i}0s"; UP=1; break
    fi
    sleep 10
done
if [ "$UP" -ne 1 ]; then
    echo "[rollout] FAILED to start within 30min"
    kill $ROLLOUT_PID 2>/dev/null
    exit 1
fi
if ! kill -0 $ROLLOUT_PID 2>/dev/null; then
    echo "[rollout] PID $ROLLOUT_PID dead; aborting"
    exit 1
fi

# -------- Launch train on NODES[1], NODES[2] (2 tasks, 4 GPUs each) --------
srun --jobid=$SLURM_JOB_ID --overlap -w "$TRAIN_A,$TRAIN_B" -N2 -n2 --gres=gpu:l40s:4 \
    --output=$LOG_DIR/train-%t.log bash -c "
set -u
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
cd $REPO_DIR
source eval_test/bin/activate
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p $MODELSCOPE_CACHE/lockers
export MODELSCOPE_CACHE=$MODELSCOPE_CACHE
export HF_DATASETS_CACHE=$HF_DATASETS_CACHE
export TMPDIR=$TMPDIR
export HF_HOME=$HF_HOME
export MASTER_ADDR=$TRAIN_A MASTER_PORT=29550
export NNODES=2 NPROC_PER_NODE=4 NODE_RANK=\$SLURM_NODEID
if [ -n \"${TEACHER_DEEPSPEED:-}\" ]; then DS_TEACHER=\"--teacher_deepspeed ${TEACHER_DEEPSPEED}\"; else DS_TEACHER=\"\"; fi
if [ -n \"${ENABLE_THINKING:-}\" ]; then THINK_FLAG=\"--enable_thinking ${ENABLE_THINKING}\"; else THINK_FLAG=\"\"; fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' swift rlhf \
    --rlhf_type gkd --gkd_logits_topk 16 --use_hf true \
    --model $STUDENT_MODEL \
    --teacher_model $TEACHER_MODEL \
    --train_type full \
    --dataset $OPD_DATASET \
    --seq_kd false --lmbda 1 --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --save_strategy epoch --save_steps 1 --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 8192 --max_completion_length $MAX_COMPLETION_LENGTH \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 --dataset_num_proc 4 \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --attn_impl flash_attn --padding_free False \
    --use_vllm true --vllm_mode server \
    --vllm_server_host $ROLLOUT_NODE --vllm_server_port 8000 \
    --report_to none \$DS_TEACHER \$THINK_FLAG
"
TRAIN_EXIT=$?
echo "[train] exit=$TRAIN_EXIT"

kill $ROLLOUT_PID 2>/dev/null
wait $ROLLOUT_PID 2>/dev/null

exit $TRAIN_EXIT
