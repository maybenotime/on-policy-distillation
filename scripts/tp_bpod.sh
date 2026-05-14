# TP-BPOD: Teacher-Prefix Backward-Progressive OPD
#
# Curriculum controlled by PREFIX_* env vars (read at trainer __init__):
#   PREFIX_DIRECTION   forward | backward (default: forward; set to backward for TP-BPOD)
#   PREFIX_K           number of stages, default 5
#   PREFIX_DELTA       per-stage delta, default 0.2
#   PREFIX_Q0          backward starting ratio, default 0.8 (→ stages 0.8, 0.6, 0.4, 0.2, 0.0)
#
# Data plugin env vars (read at module import):
#   PREFIX_TEACHER         teacher column to use, default Qwen2.5-14B-Instruct
#   PREFIX_FILTER_CORRECT  1 = drop rows where teacher got the answer wrong (default 1)
#   PREFIX_FILTER_COMPLETE 1 = drop rows where teacher rollout was truncated (default 1)
#   PREFIX_DATA_PATH       override path to final_math_dataset_updated.jsonl
#   PREFIX_STUDENT_MODEL   student model id used to load the tokenizer (set by this script)
#
# Usage:
#   bash scripts/tp_bpod.sh [student_model] [teacher_model] [data_size]
#
# Prerequisite: vLLM rollout server up on port 8000 (`scripts/rollout.sh` on GPU 0).

STUDENT_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER_MODEL="${2:-Qwen/Qwen2.5-14B-Instruct}"
DATA_SIZE="${3:-20000}"
STUDENT_MODEL_TAG="${STUDENT_MODEL##*/}"
TEACHER_MODEL_TAG="${TEACHER_MODEL##*/}"

export PREFIX_DIRECTION="${PREFIX_DIRECTION:-backward}"
export PREFIX_K="${PREFIX_K:-5}"
export PREFIX_DELTA="${PREFIX_DELTA:-0.2}"
export PREFIX_Q0="${PREFIX_Q0:-0.8}"
# PREFIX_TEACHER defaults to the basename of TEACHER_MODEL so the cache column
# used for the teacher prefix matches the teacher actually used for the reverse-KL
# loss. Override only if you intentionally want a different prefix source.
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

OUTPUT_DIR="${OUTPUT_BASE_DIR}/tp_bpod_${STUDENT_MODEL_TAG}_teacher_${TEACHER_MODEL_TAG}_K${PREFIX_K}_delta${PREFIX_DELTA}_q0${PREFIX_Q0}_filter${PREFIX_FILTER_CORRECT}_data${DATA_SIZE}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
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
    --num_train_epochs "$PREFIX_K" \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --save_steps 1000 \
    --save_total_limit 2 \
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
    --teacher_deepspeed zero3 \
    --remove_unused_columns false

CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort -V | tail -1)
echo "$CHECKPOINT" > ../checkpoint_path_${SLURM_JOB_ID:-local}.txt
echo "Checkpoint saved: $CHECKPOINT"
