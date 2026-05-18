# Plain on-policy GKD (no curriculum, no prefix mask) — baseline for SP/TP comparison.
#
# Same student/teacher/data/batch/lr as sp_fpod.sh and tp_bpod.sh, just without
# the prefix-curriculum plugin. Effective batch = 2 * 2 * 8 = 32 (2-node).
#
# Args:
#   $1 = student model        (default Qwen/Qwen2.5-0.5B-Instruct)
#   $2 = teacher model        (default Qwen/Qwen2.5-14B-Instruct)
#   $3 = data size            (default 8910 — matches SP/TP runs)
#   $4 = num_train_epochs     (default 1; pass 5 for matched-epoch baseline)

STUDENT_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER_MODEL="${2:-Qwen/Qwen2.5-14B-Instruct}"
DATA_SIZE="${3:-8910}"
NUM_EPOCHS="${4:-1}"

STUDENT_MODEL_TAG="${STUDENT_MODEL##*/}"
TEACHER_MODEL_TAG="${TEACHER_MODEL##*/}"

CACHE_BASE="${CACHE_BASE:-/home/sya225/.cache}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/localscratch/sya225/output}"

# Use $CACHE_BASE directly as HF_HOME (NOT $CACHE_BASE/huggingface) so train
# shares cache with bench-eval. Old wrapper added /huggingface suffix which
# created a duplicate model tree and caused NFS flock races when a teacher
# wasn't present at both paths.
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
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"
TEACHER_DEEPSPEED_FLAG=""
if [ -n "${TEACHER_DEEPSPEED:-}" ]; then TEACHER_DEEPSPEED_FLAG="--teacher_deepspeed $TEACHER_DEEPSPEED"; fi

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
    --vllm_mode colocate \
    --sleep_level 1 \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" \
    $TEACHER_DEEPSPEED_FLAG

CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort -V | tail -1)
echo "$CHECKPOINT" > ../checkpoint_path_${SLURM_JOB_ID:-local}.txt
echo "Checkpoint saved: $CHECKPOINT"
