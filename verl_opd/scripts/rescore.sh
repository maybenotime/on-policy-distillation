#!/bin/bash
# Re-score HumanEval+ from the corrected 656-row (164 tasks x 4 samples) files.
# evalplus caches into <samples>_eval_results.json and silently reuses it, which
# is why the first re-run returned byte-identical scores despite new samples.
set -x
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/USER/hf_cache
O=/home/USER/opd/bench/sampled
rm -f "$O"/humanevalplus_*_eval_results.json
for T in Qwen3-8B Qwen3-4B opd30k-step424; do
  echo "==== HumanEval+ (n=4 avg) : $T ===="
  "$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$O/humanevalplus_${T}.jsonl" 2>&1 | tail -6
done
echo "RESCORE_DONE $(date -Is)"
