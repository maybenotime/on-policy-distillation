#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=cs-dryrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --time=0:30:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Cold Start — DRY RUN (Plan 35, Chapter 5)
# Verifies Phase 2 pipeline: Phase 1 checkpoint → rollout server → 14B teacher GKD
# Uses CS-1 checkpoint (0.5B←3B Exp1) with 8 samples, 14B teacher

set -euo pipefail

BASE=/home/cta156/scratch/on-policy-distillation
STUDENT_CKPT="${BASE}/output/student_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-3B-Instruct_data5000/v0-20260506-012137/checkpoint-417"
TEACHER=Qwen/Qwen2.5-14B-Instruct
OUTPUT_DIR="${BASE}/output/cs1_0.5B_from3B_to14B_data8_dryrun"
DATA_SIZE=8
DATA_PATH="${BASE}/data/prompts_math_swift.jsonl"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export HF_HOME=/home/cta156/.cache/huggingface
export MODELSCOPE_CACHE=/home/cta156/.cache/modelscope
export TORCH_HOME=/home/cta156/.cache/torch
export VLLM_CACHE_ROOT=/home/cta156/.cache/vllm
export TMPDIR=/home/cta156/.cache/tmp
export HF_DATASETS_CACHE=/home/cta156/.cache/huggingface/datasets
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"

huggingface-cli login --token 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

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint saved: $CHECKPOINT"
    else
        echo "No checkpoint found at exit" >&2
    fi
}
trap save_checkpoint EXIT

echo "=== Cold Start Dry Run ==="
echo "Phase 1 checkpoint : $STUDENT_CKPT"
echo "Phase 2 teacher    : $TEACHER"
echo "Data size          : $DATA_SIZE (dry run)"
echo "Output dir         : $OUTPUT_DIR"
echo ""

# Start rollout server with Phase 1 checkpoint
CUDA_VISIBLE_DEVICES=0 swift rollout \
    --model "$STUDENT_CKPT" \
    --vllm_data_parallel_size 1 \
    --use_hf true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --padding_free true \
    --host 0.0.0.0 \
    --port 8000 \
    --load_from_cache_file false &
ROLLOUT_PID=$!
sleep 30

echo "Start Phase 2 GKD training (dry run)"
CUDA_VISIBLE_DEVICES=1,2,3 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --use_hf true \
    --model "$STUDENT_CKPT" \
    --teacher_model "$TEACHER" \
    --train_type full \
    --dataset "${DATA_PATH}#${DATA_SIZE}" \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --save_steps 80 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 8192 \
    --max_completion_length 4096 \
    --report_to none \
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
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000
echo "Dry run training finished"

kill $ROLLOUT_PID
wait $ROLLOUT_PID 2>/dev/null
