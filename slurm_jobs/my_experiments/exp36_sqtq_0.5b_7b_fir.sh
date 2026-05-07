#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=exp36-sqtq-0.5b-7b
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

# S√T√ Controlled Experiment — 0.5B ← 7B (Plan 36, Exp 1 control)
# Fixes student at 0.5B, reduces teacher gap to 7B.
# Comparing with exp36_sqtq_0.5b_14b shows whether failure scales with capacity gap.
#
# Pipeline: label → filter S√T√ → retrain GKD (all on one node)

set -euo pipefail

BASE=/home/cta156/scratch/on-policy-distillation
STUDENT=Qwen/Qwen2.5-0.5B-Instruct
TEACHER=Qwen/Qwen2.5-7B-Instruct
export LABELS_FILE="${BASE}/data/sqtq_labels_0.5b_7b.jsonl"
export FILTERED_FILE="${BASE}/data/prompts_sqtq_0.5b_7b.jsonl"

cd "$BASE"
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

# ── Step 1: Label correctness (GPU 0 only) ──
echo "=== Step 1: Labeling 5000 prompts (0.5B student, 7B teacher) ==="
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
echo "=== Step 3: GKD training ($SQTQ_SIZE samples, 0.5B ← 7B) ==="

save_checkpoint() {
    OUTPUT_BASE="${BASE}/output/student_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-7B-Instruct_data${SQTQ_SIZE}"
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
