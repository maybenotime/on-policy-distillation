#!/bin/bash
# Late-Intervention / Answer-Conditioned OPSD launcher.
#
# This is a runnable 5k prompt-level approximation. The dataset plugin conditions
# the teacher copy on the correct final answer and LI_OPSD_RATIO instruction.
# Strict suffix-only KL masking still requires a trainer patch.

set -euo pipefail

cd ..

DATA_SIZE="${DATA_SIZE:-5000}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_TAG="${MODEL##*/}"
LI_OPSD_RATIO="${LI_OPSD_RATIO:-70}"
export LI_OPSD_RATIO
export LI_OPSD_DATA_PATH="${LI_OPSD_DATA_PATH:-data/li_opsd_answer_first5k.jsonl}"

export WANDB_API_KEY="8b34bbf9238dfc1c55c378dc0dea31921d5f420b"
export WANDB_PROJECT="on_policy_self_distill"

OUTPUT="${OUTPUT:-output/li_opsd_${MODEL_TAG}_ratio${LI_OPSD_RATIO}_5k}"

echo "LI_OPSD_RATIO      = $LI_OPSD_RATIO"
echo "LI_OPSD_DATA_PATH  = $LI_OPSD_DATA_PATH"
echo "DATA_SIZE          = $DATA_SIZE"
echo "MODEL              = $MODEL"
echo "Output dir         = $OUTPUT"
echo "SLEEP_LEVEL        = ${SLEEP_LEVEL:-0}"
echo "SAVE_STEPS         = ${SAVE_STEPS:-50}"
echo "SAVE_TOTAL_LIMIT   = ${SAVE_TOTAL_LIMIT:-7}"
echo "NOTE: this is prompt-level LI-OPSD approximation, not suffix-only KL masking."

export MASTER_PORT="${MASTER_PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 10000))}"
echo "MASTER_PORT        = $MASTER_PORT"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type gkd \
    --model "$MODEL" \
    --teacher_model "$MODEL" \
    --use_hf true \
    --external_plugins source/li_opsd_plugin.py \
    --dataset "local-li-opsd-math#${DATA_SIZE}" \
    --tuner_type full \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_UTIL:-0.4}" \
    --vllm_max_model_len 2048 \
    --sleep_level "${SLEEP_LEVEL:-0}" \
    --lmbda 1.0 \
    --beta 1.0 \
    --temperature 1.0 \
    --sft_alpha "${SFT_ALPHA:-0}" \
    --torch_dtype bfloat16 \
    --max_steps "${MAX_STEPS:-320}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --save_steps "${SAVE_STEPS:-50}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-7}" \
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
