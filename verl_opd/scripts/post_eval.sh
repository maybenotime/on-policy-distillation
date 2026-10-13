#!/bin/bash
# Post-training eval of the distilled Qwen3-4B, same protocol as the baselines:
# NON-THINKING chat template, greedy, n=1, max_tokens 6144.
#
# verl saves FSDP-sharded checkpoints; vLLM needs HF format, hence the
# model_merger step. The merger writes weights+config but not always the
# tokenizer, so we copy the tokenizer files from the base model.
set -x
CKPT_ROOT=${CKPT_ROOT:-/localscratch/USER/checkpoints/opd_code_run}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
E=/home/USER/opd/envs/code_eval_env
V=/home/USER/opd/envs/verl_opd
export PATH="$E/bin:$PATH"
export HF_HOME=/localscratch/USER/hf_cache
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
export PYTHONPATH=/home/USER/opd/LiveCodeBench

# NB: sort on the trailing step number only. "sort -t_ -k3 -n" broke because
# the parent dir (opd_code_run) also contains underscores, so field 3 was
# "run/global" and the sort degraded to lexicographic: 100,150,155,50 -> 50.
STEP_NUM=$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | sed "s|.*/global_step_||" | sort -n | tail -1)
STEP_DIR="$CKPT_ROOT/global_step_$STEP_NUM"
[ -z "$STEP_DIR" ] && { echo "NO CHECKPOINT under $CKPT_ROOT"; exit 1; }
TAG="opd-$(basename "$STEP_DIR")"
echo "[post_eval] selected checkpoint: $STEP_DIR"
HF_OUT=/localscratch/USER/checkpoints/hf_${TAG}
echo "=== merging $STEP_DIR -> $HF_OUT ==="

"$V/bin/python" -m verl.model_merger merge --backend fsdp \
    --local_dir "$STEP_DIR/actor" --target_dir "$HF_OUT" 2>&1 | tail -6

# tokenizer/config files the merger may not emit
"$E/bin/python" - <<PY
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("$BASE_MODEL")
t.save_pretrained("$HF_OUT")
print("tokenizer saved to $HF_OUT")
PY
ls "$HF_OUT" | head -20

OUT=/home/USER/opd/bench/post
mkdir -p "$OUT"
cd /home/USER/opd/LiveCodeBench
for BENCH in humanevalplus lcb_v6; do
  echo "######## GEN $TAG / $BENCH  $(date -Is) ########"
  "$E/bin/python" /home/USER/opd/scripts/gen2.py \
      --model "$HF_OUT" --bench "$BENCH" --out_dir "$OUT" --tag "$TAG" \
      --max_tokens 6144 --temperature 0.0 --n 1 --tp 1 --gpu_mem_util 0.85 \
      2>&1 | grep -aE "\[gen\]|Error|Traceback|assert" | tail -10
done

echo "######## SCORING $(date -Is) ########"
echo "==== HumanEval+ : $TAG ===="
"$E/bin/python" -m evalplus.evaluate --dataset humaneval \
    --samples "$OUT/humanevalplus_${TAG}.jsonl" 2>&1 | tail -8
echo "==== LiveCodeBench v6 : $TAG ===="
cd /home/USER/opd/LiveCodeBench
"$E/bin/python" -m lcb_runner.runner.custom_evaluator \
    --custom_output_file "$OUT/lcb_v6_${TAG}.json" \
    --release_version release_v6 --start_date 2025-01-01 \
    --scenario codegeneration --num_process_evaluate 16 --timeout 8 2>&1 | tail -8
echo "POST_EVAL_DONE $(date -Is)"
