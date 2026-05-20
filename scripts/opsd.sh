#!/bin/bash
# On-Policy Self-Distillation (OPSD) — student=teacher (same weights),
# teacher sees the problem + a correct reference solution as privileged context.
#
# Args:
#   $1 = student/teacher model   (default Qwen/Qwen2.5-0.5B-Instruct)
#   $2 = data size               (default 840)
#   $3 = num_train_epochs        (default unused; we use max_steps unless EPOCHS_MODE=1)
#
# Mode selection (env, before calling this script):
#   OPSD_MODE=single (default)      — use one teacher model's correct answer
#     OPSD_TEACHER_MODEL=deepseek-r1 (default)
#   OPSD_MODE=all_correct           — concatenate all correct teachers' answers
#   OPSD_MODE=multi                 — concatenate from OPSD_TEACHER_MODELS (csv)
#
# Other env (optional):
#   OPSD_DATA_PATH                  defaults to ../data/math_all_teachers_correct.jsonl
#   PER_DEVICE_BS                   default 1
#   GRAD_ACCUM                      default 16
#   MAX_LENGTH                      default 2048
#   MAX_COMPLETION_LENGTH           default 1024
#   OPSD_LR                         default 2e-5
#   VLLM_GPU_MEM_UTIL               default 0.4
#   EPOCHS_MODE=1                   if set, uses --num_train_epochs $3 instead of --max_steps
#   OPSD_MAX_STEPS                  default 200 (only used when EPOCHS_MODE unset)
#   OUTPUT_BASE_DIR                 default /localscratch/sya225/output
#
# Usage:
#   bash opsd.sh                                                 # default 0.5B, deepseek-r1 teacher rationale
#   bash opsd.sh Qwen/Qwen2.5-3B-Instruct 7000                   # 3B student/teacher, 7k subset
#   OPSD_MODE=single OPSD_TEACHER_MODEL=Qwen2.5-14B-Instruct bash opsd.sh Qwen/Qwen2.5-0.5B-Instruct 7000

cd ..

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_SIZE="${2:-${DATA_SIZE:-840}}"
NUM_EPOCHS="${3:-1}"

export OPSD_MODE="${OPSD_MODE:-single}"
export OPSD_TEACHER_MODEL="${OPSD_TEACHER_MODEL:-deepseek-r1}"
export OPSD_TEACHER_MODELS="${OPSD_TEACHER_MODELS:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-8b34bbf9238dfc1c55c378dc0dea31921d5f420b}"
export WANDB_PROJECT="${WANDB_PROJECT:-on_policy_self_distill}"

MODEL_TAG="${MODEL##*/}"
export OPSD_DATA_PATH="${OPSD_DATA_PATH:-../data/math_all_teachers_correct.jsonl}"

# Build a short teacher tag for the output dir
if [ "$OPSD_MODE" = "multi" ]; then
    TEACHER_TAG="multi_$(echo "$OPSD_TEACHER_MODELS" | tr ',' '_' | tr -d ' ')"
else
    TEACHER_TAG="${OPSD_TEACHER_MODEL}"
fi
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/localscratch/sya225/output}"
OUTPUT_TAG_SUFFIX="${OUTPUT_TAG_SUFFIX:-data${DATA_SIZE}}"
OUTPUT="${OUTPUT_BASE_DIR}/opsd_${MODEL_TAG}_mode${OPSD_MODE}_teacher${TEACHER_TAG}_${OUTPUT_TAG_SUFFIX}"

PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
OPSD_LR="${OPSD_LR:-2e-5}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.4}"

# Step-count or epoch-mode training
if [ "${EPOCHS_MODE:-0}" = "1" ]; then
    STEPS_ARG="--num_train_epochs $NUM_EPOCHS"
else
    OPSD_MAX_STEPS="${OPSD_MAX_STEPS:-200}"
    STEPS_ARG="--max_steps $OPSD_MAX_STEPS"
fi

echo "MODEL              = $MODEL"
echo "OPSD_MODE          = $OPSD_MODE"
echo "OPSD_TEACHER_MODEL = $OPSD_TEACHER_MODEL"
echo "OPSD_TEACHER_MODELS= $OPSD_TEACHER_MODELS"
echo "OPSD_DATA_PATH     = $OPSD_DATA_PATH"
echo "Output dir         = $OUTPUT"
echo "STEPS_ARG          = $STEPS_ARG"
echo "PER_DEVICE_BS      = $PER_DEVICE_BS  GRAD_ACCUM = $GRAD_ACCUM"
echo "MAX_LENGTH         = $MAX_LENGTH  MAX_COMPLETION_LENGTH = $MAX_COMPLETION_LENGTH"
echo "LR                 = $OPSD_LR  vllm_mem = $VLLM_GPU_MEM_UTIL"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model "$MODEL" \
    --teacher_model "$MODEL" \
    --use_hf true \
    --external_plugins source/opsd_plugin.py \
    --dataset "local-opsd-math#${DATA_SIZE}" \
    --tuner_type full \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" \
    --vllm_max_model_len "$MAX_LENGTH" \
    --sleep_level 1 \
    --lmbda 1.0 \
    --beta 1.0 \
    --temperature 1.0 \
    --sft_alpha 0 \
    --torch_dtype bfloat16 \
    $STEPS_ARG \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$OPSD_LR" \
    --warmup_ratio 0.05 \
    --save_strategy epoch \
    --save_steps 1 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length "$MAX_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --save_only_model false \
    --gradient_checkpointing true \
    --use_logits_to_keep false \
    --padding_free false \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --output_dir "$OUTPUT" \
    --report_to wandb

CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
echo "$CHECKPOINT" > ../checkpoint_path_${SLURM_JOB_ID:-local}.txt
echo "Checkpoint saved: $CHECKPOINT"
