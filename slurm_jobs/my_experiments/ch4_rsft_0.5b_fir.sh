#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=exp38-rsft-0.5b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --time=5:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Ablation 2 — RSFT: SFT on student's own correct rollouts, 0.5B
# Step 1: generate student rollouts on 840 problems (8 samples each),
#         filter to correct ones, save as rsft_student_correct.jsonl
# Step 2: SFT on filtered data, 200 steps (matches OPSD exp37 step count)
#
# Expected coverage: ~30-50% of 840 problems → ~250-420 correct samples
# Effective batch = 1 (per_device) × 4 (GPUs) × 4 (accumulation) = 16

set -euo pipefail

BASE=/home/cta156/scratch/on-policy-distillation
RSFT_DATA="${BASE}/data/rsft_student_correct.jsonl"
MODEL=Qwen/Qwen2.5-0.5B-Instruct
OUTPUT="${BASE}/output/rsft_Qwen2.5-0.5B-Instruct"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

echo "========================================"
echo "Step 1: Generate student correct rollouts"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python source/generate_rsft_data.py \
    --data_path "${BASE}/data/math_all_teachers_correct.jsonl" \
    --model "$MODEL" \
    --output_path "$RSFT_DATA" \
    --n_samples 8 \
    --temperature 0.8 \
    --max_tokens 2048 \
    --gpu_memory_utilization 0.85

RSFT_SIZE=$(wc -l < "$RSFT_DATA")
echo "RSFT data: ${RSFT_SIZE} correct samples → ${RSFT_DATA}"

echo "========================================"
echo "Step 2: SFT on student correct rollouts"
echo "========================================"

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint: $CHECKPOINT"
    else
        echo "No checkpoint found" >&2
    fi
}
trap save_checkpoint EXIT

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model "$MODEL" \
    --use_hf true \
    --dataset "$RSFT_DATA" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --deepspeed zero2 \
    --max_steps 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --save_steps 50 \
    --save_total_limit 4 \
    --logging_steps 1 \
    --max_length 2048 \
    --save_only_model true \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --output_dir "$OUTPUT" \
    --report_to none

echo "RSFT training done."
