# SP-FPOD: Student-Prefix Forward-Progressive OPD
#
# Curriculum controlled by PREFIX_* env vars (read at trainer __init__):
#   PREFIX_DIRECTION   forward (default) | backward
#   PREFIX_K           number of stages, default 5
#   PREFIX_DELTA       per-stage delta, default 0.2
#   PREFIX_P0          forward starting ratio, default 0.2
#   PREFIX_Q0          backward starting ratio, default 0.8
#
# Usage:
#   bash scripts/sp_fpod.sh [student_model] [teacher_model] [data_size]

STUDENT_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER_MODEL="${2:-Qwen/Qwen2.5-14B-Instruct}"
DATA_SIZE="${3:-20000}"
STUDENT_MODEL_TAG="${STUDENT_MODEL##*/}"
TEACHER_MODEL_TAG="${TEACHER_MODEL##*/}"

export PREFIX_DIRECTION="${PREFIX_DIRECTION:-forward}"
export PREFIX_K="${PREFIX_K:-5}"
export PREFIX_DELTA="${PREFIX_DELTA:-0.2}"
export PREFIX_P0="${PREFIX_P0:-0.2}"
export PREFIX_Q0="${PREFIX_Q0:-0.8}"

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

OUTPUT_DIR="${OUTPUT_BASE_DIR}/sp_fpod_${STUDENT_MODEL_TAG}_teacher_${TEACHER_MODEL_TAG}_K${PREFIX_K}_delta${PREFIX_DELTA}_p0${PREFIX_P0}_data${DATA_SIZE}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --use_hf true \
    --model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --train_type full \
    --dataset "../data/prompts_math_swift.jsonl#${DATA_SIZE}" \
    --external_plugins ../source/prefix_curriculum_plugin.py \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs "$PREFIX_K" \
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
