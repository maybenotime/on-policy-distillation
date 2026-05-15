#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=exp36-sqtq-0.5b-14b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --time=9:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# S√T√ Controlled Experiment — FULL RUN (Plan 36, Exp 1)
# Student: Qwen2.5-0.5B-Instruct  Teacher: Qwen2.5-14B-Instruct
#
# Pipeline (all on one node):
#   Step 1 (GPU 0):   split_by_correctness.py — label 5000 prompts
#   Step 2 (CPU):     filter S√T√ subset inline
#   Step 3 (GPU 0-3): GKD re-training on S√T√ subset
#
# Time estimate: ~2h labeling + ~4h GKD = ~6h; 9h budget includes headroom.
#
# Data note: uses --dataset math --max_samples 5000 for labeling.
# If prompts_math_swift.jsonl has "answer" fields, replace with:
#   --dataset custom --data_path "${BASE}/data/prompts_math_swift.jsonl"
# to label the exact same 5000 prompts used in the original GKD training.

set -euo pipefail

BASE=/home/cta156/scratch/on-policy-distillation
STUDENT=Qwen/Qwen2.5-0.5B-Instruct
TEACHER=Qwen/Qwen2.5-14B-Instruct
export LABELS_FILE="${BASE}/data/sqtq_labels_0.5b_14b.jsonl"
export FILTERED_FILE="${BASE}/data/prompts_sqtq_0.5b_14b.jsonl"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

# ── Step 1: Label correctness (GPU 0 only) ──
echo "=== Step 1: Labeling 5000 prompts (math dataset, temp=0) ==="
CUDA_VISIBLE_DEVICES=0 python source/split_by_correctness.py \
    --dataset math \
    --max_samples 5000 \
    --student_model "$STUDENT" \
    --teacher_model "$TEACHER" \
    --output_path "$LABELS_FILE" \
    --temperature 0.0 \
    --max_new_tokens 4096

echo "Labeling done → $LABELS_FILE"

# ── Step 2: Filter S√T√ (CPU-only) ──
echo ""
echo "=== Step 2: Filtering S√T√ subset ==="
python - <<'PYEOF'
import json, os

labels_file = os.environ["LABELS_FILE"]
filtered_file = os.environ["FILTERED_FILE"]

with open(labels_file) as f:
    data = [json.loads(l) for l in f if l.strip()]

sqtq = [d for d in data if d["student_correct"] and d["teacher_correct"]]
total = len(data)
print(f"  Total labeled: {total}")
print(f"  S√T√: {len(sqtq)} ({100*len(sqtq)/total:.1f}%)")
print(f"  S✓T✗: {sum(d['student_correct'] and not d['teacher_correct'] for d in data)} ({100*sum(d['student_correct'] and not d['teacher_correct'] for d in data)/total:.1f}%)")
print(f"  S✗T✓: {sum(not d['student_correct'] and d['teacher_correct'] for d in data)} ({100*sum(not d['student_correct'] and d['teacher_correct'] for d in data)/total:.1f}%)")
print(f"  S✗T✗: {sum(not d['student_correct'] and not d['teacher_correct'] for d in data)} ({100*sum(not d['student_correct'] and not d['teacher_correct'] for d in data)/total:.1f}%)")

assert len(sqtq) > 0, "No S√T√ samples found — aborting"

with open(filtered_file, "w") as f:
    for d in sqtq:
        f.write(json.dumps({"messages": d["messages"]}, ensure_ascii=False) + "\n")

print(f"  Written {len(sqtq)} S√T√ prompts → {filtered_file}")
PYEOF

SQTQ_SIZE=$(wc -l < "$FILTERED_FILE")
echo "S√T√ subset size: $SQTQ_SIZE"

# ── Step 3: GKD training on S√T√ subset ──
echo ""
echo "=== Step 3: GKD training on S√T√ subset ($SQTQ_SIZE samples) ==="

save_checkpoint() {
    OUTPUT_BASE="${BASE}/output/student_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-14B-Instruct_data${SQTQ_SIZE}_sqtq"
    CHECKPOINT=$(find "$OUTPUT_BASE" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "${BASE}/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint: $CHECKPOINT"
    else
        echo "No checkpoint found" >&2
    fi
}
trap save_checkpoint EXIT

cd "${BASE}/scripts"

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash rollout.sh "$STUDENT" &
ROLLOUT_PID=$!
sleep 30

echo "GKD start"
CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 bash on-policy-distill.sh \
    "$STUDENT" \
    "$TEACHER" \
    "$SQTQ_SIZE" \
    "$FILTERED_FILE"
echo "GKD done"

kill $ROLLOUT_PID
wait $ROLLOUT_PID 2>/dev/null
