#!/bin/bash
# Baselines: Qwen3-8B (teacher) + Qwen3-4B (student), NON-THINKING,
# HumanEval+ (164) and LiveCodeBench v6 start_date=2025-01-01 (182). Greedy, n=1.
set -x
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH"
export HF_HOME=/localscratch/USER/hf_cache
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
export PYTHONPATH=/home/USER/opd/LiveCodeBench
OUT=/home/USER/opd/bench/baseline
mkdir -p "$OUT"
cd /home/USER/opd/LiveCodeBench

for MODEL in Qwen/Qwen3-8B Qwen/Qwen3-4B; do
  TAG=$(basename "$MODEL")
  for BENCH in humanevalplus lcb_v6; do
    echo "######## GEN $TAG / $BENCH  $(date -Is) ########"
    "$E/bin/python" /home/USER/opd/scripts/gen2.py \
        --model "$MODEL" --bench "$BENCH" --out_dir "$OUT" --tag "$TAG" \
        --max_tokens 6144 --temperature 0.0 --n 1 --tp 1 --gpu_mem_util 0.85 \
        2>&1 | grep -aE "\[gen\]|Error|Traceback|assert" | tail -10
  done
done

echo "######## SCORING  $(date -Is) ########"
for TAG in Qwen3-8B Qwen3-4B; do
  echo "==== HumanEval+ : $TAG ===="
  "$E/bin/python" -m evalplus.evaluate --dataset humaneval \
      --samples "$OUT/humanevalplus_${TAG}.jsonl" 2>&1 | tail -8
  echo "==== LiveCodeBench v6 : $TAG ===="
  cd /home/USER/opd/LiveCodeBench
  "$E/bin/python" -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$OUT/lcb_v6_${TAG}.json" \
      --release_version release_v6 --start_date 2025-01-01 \
      --scenario codegeneration --num_process_evaluate 16 --timeout 8 2>&1 | tail -8
done
echo "BASELINE_DONE $(date -Is)"
