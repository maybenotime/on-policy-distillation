#!/bin/bash
# On-Policy Self-Distillation (OPSD) — Qwen2.5-0.5B-Instruct student
#
# Mode selection (set before calling this script):
#   OPSD_MODE=single (default)  — use one teacher model's correct answer
#   OPSD_TEACHER_MODEL=deepseek-r1 (default)
#
#   OPSD_MODE=all_correct       — concatenate all correct teachers' answers
#
#   OPSD_MODE=multi             — concatenate correct answers from a subset
#   OPSD_TEACHER_MODELS=deepseek-r1,Qwen2.5-7B-Instruct,Llama-3.1-8B-Instruct
#
# Usage:
#   bash opsd.sh
#   OPSD_MODE=single OPSD_TEACHER_MODEL=Qwen2.5-72B-Instruct bash opsd.sh
#   OPSD_MODE=multi OPSD_TEACHER_MODELS=deepseek-r1,Qwen2.5-7B-Instruct bash opsd.sh

cd ..

export OPSD_MODE="${OPSD_MODE:-single}"
export OPSD_TEACHER_MODEL="${OPSD_TEACHER_MODEL:-deepseek-r1}"
export OPSD_TEACHER_MODELS="${OPSD_TEACHER_MODELS:-}"
DATA_SIZE="${DATA_SIZE:-840}"
export WANDB_API_KEY="8b34bbf9238dfc1c55c378dc0dea31921d5f420b"
export WANDB_PROJECT="on_policy_self_distill"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_TAG="${MODEL##*/}"
export OPSD_DATA_PATH="${OPSD_DATA_PATH:-../data/math_all_teachers_correct.jsonl}"

# Build a short teacher tag for the output dir
if [ "$OPSD_MODE" = "multi" ]; then
    TEACHER_TAG="multi_$(echo "$OPSD_TEACHER_MODELS" | tr ',' '_' | tr -d ' ')"
else
    TEACHER_TAG="${OPSD_TEACHER_MODEL}"
fi
OUTPUT="${OUTPUT:-output/opsd_${MODEL_TAG}_mode${OPSD_MODE}_teacher${TEACHER_TAG}}"

echo "OPSD_MODE          = $OPSD_MODE"
echo "OPSD_TEACHER_MODEL = $OPSD_TEACHER_MODEL"
echo "OPSD_TEACHER_MODELS= $OPSD_TEACHER_MODELS"
echo "OPSD_DATA_PATH     = $OPSD_DATA_PATH"
echo "Output dir         = $OUTPUT"

# Unique port per job to avoid EADDRINUSE when multiple OPSD jobs share a node
export MASTER_PORT="${MASTER_PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 10000))}"
echo "MASTER_PORT        = $MASTER_PORT"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
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
    --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.4}" \
    --vllm_max_model_len 2048 \
    --sleep_level 1 \
    --lmbda 1.0 \
    --beta 1.0 \
    --temperature 1.0 \
    --sft_alpha 0 \
    --torch_dtype bfloat16 \
    --max_steps "${MAX_STEPS:-200}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --save_steps 50 \
    --save_total_limit 4 \
    --logging_steps 1 \
    --max_length 2048 \
    --max_completion_length 1024 \
    --save_only_model true \
    --gradient_checkpointing true \
    --use_logits_to_keep false \
    --padding_free false \
    --attn_impl flash_attn \
    --output_dir "$OUTPUT" \
    --report_to "${REPORT_TO:-none}"
