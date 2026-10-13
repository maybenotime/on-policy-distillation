#!/bin/bash
# Score both code benchmarks for all 5 models from the v5 (8192-cap) generations.
# evalplus caches into <samples>_eval_results.json and silently reuses it, so the
# cache MUST be deleted first or scores come back byte-identical to the old run.
E=/home/sya225/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache
O=/home/sya225/opd/bench/official
rm -f "$O"/*_eval_results.json "$O"/*_output_eval.json "$O"/*_codegeneration_output_eval.json

for T in Qwen3-8B Qwen3-4B Qwen3-1.7B opd-4b opd-1.7b; do
  echo "######## HumanEval+ (n=4 avg pass@1) : $T ########"
  "$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$O/humanevalplus_${T}.jsonl" 2>&1 | tail -6
done
for T in Qwen3-8B Qwen3-4B Qwen3-1.7B opd-4b opd-1.7b; do
  echo "######## LCB v6 : $T ########"
  "$E/bin/python" -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$O/lcb_v6_${T}.json" \
      --release_version release_v6 --start_date 2025-01-01 \
      --scenario codegeneration --num_process_evaluate 16 --timeout 8 2>&1 | tail -8
done
echo "SCORE_V5_DONE $(date -Is)"
