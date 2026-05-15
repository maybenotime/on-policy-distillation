#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=exp38-rsft-dryrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# RSFT dry run — 0.5B student, 20 problems, 5 SFT steps
# Verifies: generate_rsft_data.py runs, math_verify works,
#           swift sft loads local JSONL, loss is non-zero

set -euo pipefail

BASE=/home/cta156/scratch/on-policy-distillation
MODEL=Qwen/Qwen2.5-0.5B-Instruct
RSFT_DATA="${BASE}/data/rsft_dryrun.jsonl"
OUTPUT="${BASE}/output/rsft_dryrun"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

echo "========================================"
echo "Step 1: Generate rollouts (20 problems)"
echo "========================================"

# Use head to take only 20 problems for the dry run
DRYRUN_DATA="${BASE}/data/rsft_dryrun_input.jsonl"
head -n 20 "${BASE}/data/math_all_teachers_correct.jsonl" > "$DRYRUN_DATA"

CUDA_VISIBLE_DEVICES=0 python source/generate_rsft_data.py \
    --data_path "$DRYRUN_DATA" \
    --model "$MODEL" \
    --output_path "$RSFT_DATA" \
    --n_samples 8 \
    --temperature 0.8 \
    --max_tokens 2048 \
    --gpu_memory_utilization 0.85

RSFT_SIZE=$(wc -l < "$RSFT_DATA")
echo "Generated ${RSFT_SIZE} correct samples"

if [ "$RSFT_SIZE" -eq 0 ]; then
    echo "ERROR: No correct rollouts generated from 20 problems" >&2
    exit 1
fi

echo "========================================"
echo "Step 2: SFT dry run (5 steps)"
echo "========================================"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model "$MODEL" \
    --use_hf true \
    --dataset "$RSFT_DATA" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --deepspeed zero2 \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --max_length 2048 \
    --save_only_model true \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --output_dir "$OUTPUT" \
    --report_to none

echo "Dry run complete."
