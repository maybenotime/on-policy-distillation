#!/bin/bash
# Run OOD analysis (recovery rate + per-token PPL/entropy/correctness) on
# every per-epoch checkpoint of a run.
#
# Args:
#   $1 = run_output_dir   (path containing checkpoint-* subdirs)
#   $2 = teacher_model    (live teacher for completion accuracy)
#   $3 = pinned_jsonl     (default data/pinned_eval_64.jsonl)
#
# For each `checkpoint-N`, calls source/ood_analysis.py and writes
# results/pinned_curves/<run_tag>/<checkpoint>.pkl.

set -e

RUN_DIR="${1:?Usage: $0 <run_output_dir> <teacher_model> [pinned_jsonl]}"
TEACHER_MODEL="${2:?missing teacher_model}"
PINNED_JSONL="${3:-data/pinned_eval_64.jsonl}"

RUN_TAG=$(basename "$RUN_DIR")
OUT_BASE="results/pinned_curves/${RUN_TAG}"
mkdir -p "$OUT_BASE"

echo "[pinned_eval] run_dir=$RUN_DIR  teacher=$TEACHER_MODEL  pinned=$PINNED_JSONL"

for CKPT in $(find "$RUN_DIR" -maxdepth 2 -type d -name "checkpoint-*" | sort -V); do
    CKPT_NAME=$(basename "$CKPT")
    OUT="${OUT_BASE}/${CKPT_NAME}.pkl"
    if [ -f "$OUT" ]; then
        echo "[pinned_eval] $CKPT_NAME already done, skipping"
        continue
    fi
    echo "[pinned_eval] evaluating $CKPT_NAME -> $OUT"
    python source/ood_analysis.py \
        --student_model "$CKPT" \
        --teacher_model "$TEACHER_MODEL" \
        --prompts_file "$PINNED_JSONL" \
        --output_path "$OUT" \
        --num_rollouts 4 \
        --num_sample_positions 10 \
        --use_vllm
done

echo "[pinned_eval] done. Outputs in $OUT_BASE"
