# Q2.5 OPD with separate vLLM rollout server (--vllm_mode server) for speed.
# Mirror of opd_baseline_qwen3_server.sh but without --enable_thinking (Q2.5 has
# no thinking mode).
#
# Args:
#   $1 = student model        (default Qwen/Qwen2.5-0.5B-Instruct)
#   $2 = teacher model        (default Qwen/Qwen2.5-32B-Instruct)
#   $3 = data size            (default 10000)
#   $4 = num_train_epochs     (default 1)
#
# Required env (set by wrapper):
#   CACHE_BASE, OUTPUT_BASE_DIR, VLLM_SERVER_PORT

STUDENT_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER_MODEL="${2:-Qwen/Qwen2.5-32B-Instruct}"
DATA_SIZE="${3:-10000}"
NUM_EPOCHS="${4:-1}"

STUDENT_MODEL_TAG="${STUDENT_MODEL##*/}"
TEACHER_MODEL_TAG="${TEACHER_MODEL##*/}"

CACHE_BASE="${CACHE_BASE:-/home/sya225/scratch/hf_cache}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/home/sya225/scratch/checkpoints}"

export HF_HOME="${HF_HOME:-$CACHE_BASE}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_BASE/modelscope}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm}"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"

export WANDB_API_KEY="8b34bbf9238dfc1c55c378dc0dea31921d5f420b"
export WANDB_PROJECT="${WANDB_PROJECT:-prefix_curriculum_opd}"

OUTPUT_TAG_SUFFIX="${OUTPUT_TAG_SUFFIX:-data${DATA_SIZE}}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/opd_baseline_${STUDENT_MODEL_TAG}_teacher_${TEACHER_MODEL_TAG}_epochs${NUM_EPOCHS}_${OUTPUT_TAG_SUFFIX}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_SPEC="${OPD_DATASET:-${REPO_DIR}/data/prompts_math_swift.jsonl#${DATA_SIZE}}"

VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-3}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --gkd_logits_topk 16 \
    --use_hf true \
    --model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --train_type full \
    --dataset "$DATASET_SPEC" \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --save_strategy epoch \
    --save_steps 1 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 8192 \
    --max_completion_length 4096 \
    --report_to wandb \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --save_only_model false \
    --dataloader_num_workers 16 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --padding_free False \
    --use_logits_to_keep false \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port "$VLLM_SERVER_PORT"

CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort -V | tail -1)
echo "$CHECKPOINT" > ../checkpoint_path_${SLURM_JOB_ID:-local}.txt
echo "Checkpoint saved: $CHECKPOINT"
