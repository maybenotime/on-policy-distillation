#!/bin/bash
E=/home/sya225/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache
O=/home/sya225/opd/bench/official
rm -f "$O"/humanevalplus_*_eval_results.json     # evalplus silently reuses this
for T in Qwen3-8B Qwen3-4B opd-4b Qwen3-1.7B opd-1.7b opd-1.7b-from14b; do
  echo "######## $T ########"
  "$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$O/humanevalplus_${T}.jsonl" 2>&1 | grep -aE "^pass@1|humaneval"
done
echo "HE_SCORE_DONE $(date -Is)"
