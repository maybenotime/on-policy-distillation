# TP-BPOD for Qwen3 (thinking mode)
#
# Note: the cached Qwen3-14B trajectories in final_math_dataset_updated.jsonl
# contain <think>...</think> reasoning blocks, so the student must run in
# thinking mode to continue them coherently. This is intentional — Qwen3
# non-thinking TP-BPOD is not supported here.
#
# Differences from tp_bpod.sh:
#   - default student/teacher = Qwen3-1.7B / Qwen3-14B
#   - --enable_thinking true on rollout + trainer
#   - per_device=1, grad_accum=4 (1.7B student bigger than 0.5B)
#   - vllm_util 0.3 (tighter — thinking-mode KV cache is much longer)
#   - max_length 16384, max_completion_length 8192 (thinking blocks are long)
#   - PREFIX_TEACHER default = Qwen3-14B
#
# Curriculum + filter env vars:
#   PREFIX_DIRECTION       backward (default for tp)
#   PREFIX_K               number of stages, default 10 (step-based within 1 epoch)
#   PREFIX_DELTA           per-stage delta, default 0.1
#   PREFIX_Q0              starting ratio, default 0.9 (stages 0.9, 0.8, ..., 0.0)
#   PREFIX_FILTER_CORRECT  default 1
#   PREFIX_FILTER_COMPLETE default 1

STUDENT_MODEL="${1:-Qwen/Qwen3-1.7B}"
TEACHER_MODEL="${2:-Qwen/Qwen3-14B}"
DATA_SIZE="${3:-9000}"
STUDENT_MODEL_TAG="${STUDENT_MODEL##*/}"
TEACHER_MODEL_TAG="${TEACHER_MODEL##*/}"

export PREFIX_DIRECTION="${PREFIX_DIRECTION:-backward}"
export PREFIX_K="${PREFIX_K:-10}"
export PREFIX_DELTA="${PREFIX_DELTA:-0.1}"
export PREFIX_Q0="${PREFIX_Q0:-0.9}"
export PREFIX_TEACHER="${PREFIX_TEACHER:-$TEACHER_MODEL_TAG}"
export PREFIX_FILTER_CORRECT="${PREFIX_FILTER_CORRECT:-1}"
export PREFIX_FILTER_COMPLETE="${PREFIX_FILTER_COMPLETE:-1}"
export PREFIX_STUDENT_MODEL="${PREFIX_STUDENT_MODEL:-$STUDENT_MODEL}"

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

OUTPUT_DIR="${OUTPUT_BASE_DIR}/tp_bpod_qwen3_${STUDENT_MODEL_TAG}_teacher_${TEACHER_MODEL_TAG}_K${PREFIX_K}_delta${PREFIX_DELTA}_q0${PREFIX_Q0}_filter${PREFIX_FILTER_CORRECT}_data${DATA_SIZE}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --enable_thinking true \
    --use_hf true \
    --model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --train_type full \
    --dataset "prefix-curriculum-backward-math#${DATA_SIZE}" \
    --external_plugins ../source/prefix_curriculum_plugin.py \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 0.1 \
    --save_total_limit 11 \
    --logging_steps 1 \
    --max_length 16384 \
    --max_completion_length 8192 \
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
    --vllm_gpu_memory_utilization 0.3 \
    --teacher_deepspeed zero3 \
    --remove_unused_columns false

CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort -V | tail -1)
echo "$CHECKPOINT" > ../checkpoint_path_${SLURM_JOB_ID:-local}.txt
echo "Checkpoint saved: $CHECKPOINT"
