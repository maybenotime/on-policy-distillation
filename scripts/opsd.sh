#!/bin/bash
# On-Policy Self-Distillation (OPSD) — test run with Qwen2.5-0.5B-Instruct
#
# OPSD: student sees only the problem; teacher (same model weights) sees
#       problem + a correct reference solution as privileged information.
#
# Mode selection (set before calling this script):
#   OPSD_MODE=single (default)  — use one teacher model's correct answer
#   OPSD_TEACHER_MODEL=deepseek-r1 (default)
#
#   OPSD_MODE=all_correct       — concatenate all correct teachers' answers
#
# Usage:
#   bash opsd.sh                                      # defaults
#   OPSD_MODE=single OPSD_TEACHER_MODEL=Qwen2.5-72B-Instruct bash opsd.sh
#   OPSD_MODE=all_correct bash opsd.sh

cd ..

export OPSD_MODE="${OPSD_MODE:-single}"
export OPSD_TEACHER_MODEL="${OPSD_TEACHER_MODEL:-deepseek-r1}"
export WANDB_API_KEY="8b34bbf9238dfc1c55c378dc0dea31921d5f420b"
export WANDB_PROJECT="on_policy_self_distill"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
MODEL_TAG="${MODEL##*/}"
export OPSD_DATA_PATH="${OPSD_DATA_PATH:-/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl}"
OUTPUT="output/opsd_${MODEL_TAG}_mode${OPSD_MODE}_teacher${OPSD_TEACHER_MODEL}"

echo "OPSD_MODE        = $OPSD_MODE"
echo "OPSD_TEACHER_MODEL = $OPSD_TEACHER_MODEL"
echo "Output dir       = $OUTPUT"

# when test, we delete the deepspeed config and use static mode (have teacher_model)
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type gkd \
    --model "$MODEL" \
    --use_hf true \
    --teacher_model "$MODEL" \
    --external_plugins source/opsd_plugin.py \
    --dataset local-opsd-math \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_max_model_len 2048 \
    --sleep_level 1 \
    --lmbda 1.0 \
    --beta 1.0 \
    --temperature 1.0 \
    --sft_alpha 0 \
    --torch_dtype bfloat16 \
    --max_steps 200 \
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
    --report_to wandb
