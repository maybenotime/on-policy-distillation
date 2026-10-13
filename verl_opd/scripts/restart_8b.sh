#!/bin/bash
# Restart only the Qwen3-8B worker with TP=2 on GPUs 0+5.
# Kill by PID rather than `pkill -f`: the pattern would also match this script's
# own srun command line and take the whole step down with it.
set -x
for pid in $(pgrep -u sya225 -f "eval_one_model.sh qwen3_8b"); do
    pkill -P "$pid" 2>/dev/null
    kill "$pid" 2>/dev/null
    echo "killed worker $pid"
done
sleep 5
# any straggler generator for this tag
for pid in $(pgrep -u sya225 -f "gen2.py --model Qwen/Qwen3-8B"); do
    kill "$pid" 2>/dev/null; echo "killed gen $pid"
done
sleep 10
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' '|'; echo

CUDA_VISIBLE_DEVICES=0,5 nohup bash /home/sya225/opd/scripts/eval_one_model.sh \
    qwen3_8b Qwen/Qwen3-8B Qwen3-8B 8010 0,5 2 \
    > /localscratch/sya225/logs/evalall_Qwen3-8B.log 2>&1 &
echo "restarted 8B with TP=2, pid=$!"
