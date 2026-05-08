# On-Policy Distillation for Qwen3 (non-thinking mode)
# Hardcoded --enable_thinking false on both rollout (separate process) and
# the swift rlhf trainer so the chat templates match.

STUDENT_MODEL="${1:-Qwen/Qwen3-1.7B}"
TEACHER_MODEL="${2:-Qwen/Qwen3-8B}"
DATA_SIZE="${3:-5000}"
DATASET_PATH="${4:-../data/prompts_math_swift.jsonl}"
STUDENT_MODEL_TAG="${STUDENT_MODEL##*/}"
TEACHER_MODEL_TAG="${TEACHER_MODEL##*/}"

CACHE_BASE="${CACHE_BASE:-/scratch/sya225/hf_cache}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/scratch/sya225/checkpoints}"
SAVE_STEPS="${SAVE_STEPS:-200}"

# Respect pre-set HF_HOME (caller may already point at an existing model cache).
export HF_HOME="${HF_HOME:-$CACHE_BASE}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_BASE/modelscope}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm}"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"

HF_TOKEN="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"
huggingface-cli login --token $HF_TOKEN

export WANDB_API_KEY="8b34bbf9238dfc1c55c378dc0dea31921d5f420b"
export WANDB_PROJECT="on_policy_distill"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --enable_thinking false \
    --use_hf true \
    --model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --train_type full \
    --dataset "${DATASET_PATH}#${DATA_SIZE}" \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 8192 \
    --max_completion_length 4096 \
    --report_to wandb \
    --output_dir "${OUTPUT_BASE_DIR}/qwen3_student_${STUDENT_MODEL_TAG}_teacher_${TEACHER_MODEL_TAG}_data${DATA_SIZE}" \
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
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000

OUTPUT_BASE="${OUTPUT_BASE_DIR}/qwen3_student_${STUDENT_MODEL_TAG}_teacher_${TEACHER_MODEL_TAG}_data${DATA_SIZE}"
CHECKPOINT=$(find "$OUTPUT_BASE" -name "checkpoint-*" -type d | sort -V | tail -1)
echo "$CHECKPOINT" > ../checkpoint_path_${SLURM_JOB_ID}.txt
echo "Checkpoint saved: $CHECKPOINT"
