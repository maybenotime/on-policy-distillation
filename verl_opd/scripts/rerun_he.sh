#!/bin/bash
# Re-run HumanEval+ only. The earlier pass generated n=4 but wrote just
# outputs[0], so the scores were single random draws at T=0.6 -- noisier than
# greedy. gen2.py now emits every sample; evalplus groups rows by task_id, so
# this yields a real avg pass@1. LCB already used all 4 samples and is unaffected.
set -x
E=/home/sya225/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache
export HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets
export PYTHONPATH=/home/sya225/opd/LiveCodeBench
OUT=/home/sya225/opd/bench/sampled
cd /home/sya225/opd/LiveCodeBench

for pair in "Qwen/Qwen3-8B:Qwen3-8B" "Qwen/Qwen3-4B:Qwen3-4B" "/localscratch/sya225/checkpoints/hf_opd30k_step424:opd30k-step424"; do
  M="${pair%%:*}"; T="${pair##*:}"
  echo "######## GEN $T / humanevalplus $(date -Is) ########"
  "$E/bin/python" /home/sya225/opd/scripts/gen2.py --model "$M" --bench humanevalplus \
      --out_dir "$OUT" --tag "$T" --max_tokens 6144 \
      --temperature 0.6 --top_p 0.95 --top_k 20 --n 4 --tp 1 --gpu_mem_util 0.85 \
      2>&1 | grep -aE "\[gen\]|rror|Traceback|unrecognized" | tail -6
done

for T in Qwen3-8B Qwen3-4B opd30k-step424; do
  echo "==== HumanEval+ n=4 avg : $T ===="
  echo -n "rows: "; wc -l < "$OUT/humanevalplus_${T}.jsonl"
  "$E/bin/python" -m evalplus.evaluate --dataset humaneval \
      --samples "$OUT/humanevalplus_${T}.jsonl" 2>&1 | tail -6
done
echo "RERUN_HE_DONE $(date -Is)"
