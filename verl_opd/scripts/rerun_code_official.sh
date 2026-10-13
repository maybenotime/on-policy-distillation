#!/bin/bash
# Re-run BOTH code benchmarks for all five models under Qwen3's OFFICIAL
# non-thinking sampling recommendation.
#
# The previous pass used T=0.6/top_p=0.95 — those are generation_config.json's
# values, which Qwen3's model card gives for *thinking* mode. For
# enable_thinking=False the card specifies T=0.7, top_p=0.8, top_k=20, min_p=0.
# max_tokens 38912 is the card's suggestion for math/programming competition
# benchmarking (we previously used 6144).
#
# Results land in bench/official/ so the older bench/sampled/ numbers are not
# silently overwritten or mixed.
set -x
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/USER/hf_cache
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
export PYTHONPATH=/home/USER/opd/LiveCodeBench
OUT=/home/USER/opd/bench/official
mkdir -p "$OUT"
cd /home/USER/opd/LiveCodeBench

SAMP="--temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --n 4 --max_tokens 32768"

for pair in \
    "Qwen/Qwen3-8B:Qwen3-8B" \
    "Qwen/Qwen3-4B:Qwen3-4B" \
    "Qwen/Qwen3-1.7B:Qwen3-1.7B" \
    "/localscratch/USER/checkpoints/hf_opd30k_step424:opd-4b" \
    "/localscratch/USER/checkpoints/hf_opd17b_step424:opd-1.7b" ; do
  M="${pair%%:*}"; T="${pair##*:}"
  [ -e "$M" ] || [[ "$M" == Qwen/* ]] || { echo "SKIP missing $M"; continue; }
  for BENCH in humanevalplus lcb_v6; do
    echo "######## GEN $T / $BENCH $(date -Is) ########"
    "$E/bin/python" /home/USER/opd/scripts/gen2.py --model "$M" --bench "$BENCH" \
        --out_dir "$OUT" --tag "$T" $SAMP --tp 1 --gpu_mem_util 0.90 \
        2>&1 | grep -aE "\[gen\]|rror|Traceback|unrecognized" | tail -6
  done
done

echo "######## SCORING $(date -Is) ########"
for T in Qwen3-8B Qwen3-4B Qwen3-1.7B opd-4b opd-1.7b; do
  [ -f "$OUT/humanevalplus_${T}.jsonl" ] || continue
  echo "==== HumanEval+ : $T ===="
  rm -f "$OUT/humanevalplus_${T}_eval_results.json"   # evalplus silently reuses this
  "$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$OUT/humanevalplus_${T}.jsonl" 2>&1|tail -5
  echo "==== LiveCodeBench v6 : $T ===="
  cd /home/USER/opd/LiveCodeBench
  "$E/bin/python" -m lcb_runner.runner.custom_evaluator --custom_output_file "$OUT/lcb_v6_${T}.json" \
      --release_version release_v6 --start_date 2025-01-01 \
      --scenario codegeneration --num_process_evaluate 16 --timeout 8 2>&1|tail -4
done
echo "RERUN_OFFICIAL_DONE $(date -Is)"
