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

export HF_HOME="${HF_HOME:-$CACHE_BASE/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_BASE/modelscope}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm}"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"

HF_TOKEN="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"
huggingface-cli login --token $HF_TOKEN

export WANDB_API_KEY="8b34bbf9238dfc1c55c378dc0dea31921d5f420b"
export WANDB_PROJECT="${WANDB_PROJECT:-prefix_curriculum_opd}"

OUTPUT_DIR="${OUTPUT_BASE_DIR}/opd_baseline_${STUDENT_MODEL_TAG}_teacher_${TEACHER_MODEL_TAG}_epochs${NUM_EPOCHS}_data${DATA_SIZE}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --use_hf true \
    --model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --train_type full \
    --dataset "../data/prompts_math_swift.jsonl#${DATA_SIZE}" \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --save_strategy epoch \
    --save_total_limit 6 \
    --logging_steps 1 \
    --max_length 8192 \
    --max_completion_length 4096 \
    --report_to wandb \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --save_only_model true \
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
    --vllm_gpu_memory_utilization 0.5 \
    --teacher_deepspeed zero3

CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort -V | tail -1)
echo "$CHECKPOINT" > ../checkpoint_path_${SLURM_JOB_ID:-local}.txt
echo "Checkpoint saved: $CHECKPOINT"
