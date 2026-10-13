#!/bin/bash
# Does the 13.7% LCB v6 length-truncation rate survive a much larger cap?
#
# At max_tokens=6144 every empty extraction had finish_reason='length', and the
# output-length distribution was bimodal (p50=296, p90=6144) -- consistent with
# non-terminating generation rather than answers needing slightly more room, but
# three order statistics cannot settle it. This reruns at 16384 and compares the
# finish_reason split directly. Teacher and student run concurrently on separate GPUs.
set -x
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH"
export HF_HOME=/localscratch/USER/hf_cache
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
export PYTHONPATH=/home/USER/opd/LiveCodeBench
cd /home/USER/opd/LiveCodeBench

CUDA_VISIBLE_DEVICES=0 "$E/bin/python" /home/USER/opd/scripts/diag_trunc.py \
    --model Qwen/Qwen3-8B --max_tokens 16384 > /localscratch/USER/logs/trunc_8b.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 "$E/bin/python" /home/USER/opd/scripts/diag_trunc.py \
    --model Qwen/Qwen3-4B --max_tokens 16384 > /localscratch/USER/logs/trunc_4b.log 2>&1 &
P2=$!
wait $P1; echo "8B_RC=$?"
wait $P2; echo "4B_RC=$?"
echo "TRUNC_TEST_DONE $(date -Is)"
