#!/bin/bash
set -x
E=/home/sya225/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache
export HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets
export PYTHONPATH=/home/sya225/opd/LiveCodeBench
cd /home/sya225/opd/LiveCodeBench
# GPU0: model default sampling (T=0.6/top_p=0.95/top_k=20)
CUDA_VISIBLE_DEVICES=0 "$E/bin/python" /home/sya225/opd/scripts/diag2.py --model Qwen/Qwen3-8B \
    --max_tokens 6144 --temperature 0.6 --top_p 0.95 --top_k 20 > /localscratch/sya225/logs/t06_8b.log 2>&1 &
P1=$!
# GPU1: greedy again, but this time dumping the text so we can SEE the failure
CUDA_VISIBLE_DEVICES=1 "$E/bin/python" /home/sya225/opd/scripts/diag2.py --model Qwen/Qwen3-8B \
    --max_tokens 6144 --temperature 0.0 --top_p 1.0 --top_k -1 > /localscratch/sya225/logs/t00_8b.log 2>&1 &
P2=$!
wait $P1; echo "T06_RC=$?"; wait $P2; echo "T00_RC=$?"
echo "TEMP_TEST_DONE $(date -Is)"
