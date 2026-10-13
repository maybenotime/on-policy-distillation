#!/bin/bash
V=/localscratch/sya225/logs/vllm_opd_1.7b_fromK4Bcode300_16k.log
echo "preemptions: $(grep -ac -i preempt $V)"; grep -ai preempt $V | tail -2 | cut -c1-220
grep -aoE "GPU KV cache usage: [0-9.]+%|Avg generation throughput: [0-9.]+" $V | tail -4 | paste - -
grep -aE "gpu.memory.utilization|max.model.len|KV cache size|Maximum concurrency" $V | head -4 | cut -c1-200
P1="post_run.sh opd-1.7b-fromK4Bcode300-16k"; P2="eval_v5.sh opd_1.7b_fromK4Bcode300_16k"; P3="scripts/run.py --comp"; P4="api_server.*--port 8033"; P5="eval300.sh"
pkill -f "$P5"; pkill -f "$P1"; pkill -f "$P2"; pkill -f "$P3"; sleep 3; pkill -f "$P4"; sleep 15
pgrep -fa "8033|run.py --comp" | grep -v pgrep | wc -l
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
