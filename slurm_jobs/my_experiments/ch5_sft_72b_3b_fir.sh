#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=sft-72b-3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=4:30:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Chapter 5: SFT baseline — 3B student, 72B teacher solutions (offline).
# Extracts Qwen2.5-72B-Instruct correct solutions from math_all_teachers_correct.jsonl
# and fine-tunes 3B with swift sft (max_steps=200 to match OPSD training budget).

set -euo pipefail

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SFT_DATA="${BASE}/data/sft_72b_correct_840.jsonl"
OUTPUT="${OUTPUT:-${BASE}/output/sft_3B_72bref}"

echo "========================================"
echo "SFT-72B 3B — extract + train"
echo "========================================"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

export HF_HOME="${HF_HOME:-/home/cta156/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/home/cta156/.cache/modelscope}"
export TORCH_HOME="${TORCH_HOME:-/home/cta156/.cache/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/home/cta156/.cache/vllm}"
export TMPDIR="${TMPDIR:-/home/cta156/.cache/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/cta156/.cache/huggingface/datasets}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR" "$HF_DATASETS_CACHE"

# Extract 72B correct solutions into swift sft format
echo "--- Extracting 72B correct solutions ---"
BASE_PATH="$BASE" python - <<'PYEOF'
import json, os

base = os.environ["BASE_PATH"]
src = f"{base}/data/math_all_teachers_correct.jsonl"
dst = f"{base}/data/sft_72b_correct_840.jsonl"
teacher = "Qwen2.5-72B-Instruct"
count = 0

with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        entry = json.loads(line)
        t = entry.get(teacher, {})
        if t.get("correctness_math_verify"):
            messages = list(entry["messages"]) + [
                {"role": "assistant", "content": t["generation"]}
            ]
            fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

print(f"Extracted {count} samples → {dst}", flush=True)
PYEOF

DATA_SIZE=$(wc -l < "$SFT_DATA")
echo "SFT data: ${DATA_SIZE} samples → ${SFT_DATA}"

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}_sft72b_3b.txt"
        echo "Checkpoint: $CHECKPOINT"
    fi
}
trap save_checkpoint EXIT

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
swift sft \
    --model "$MODEL" \
    --use_hf true \
    --dataset "$SFT_DATA" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --max_steps 200 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --save_steps 50 \
    --save_total_limit 4 \
    --logging_steps 1 \
    --max_length 2048 \
    --save_only_model true \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --output_dir "$OUTPUT" \
    --report_to none

echo "Done: SFT-72B 3B"
