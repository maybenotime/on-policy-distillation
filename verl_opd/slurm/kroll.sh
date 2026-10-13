#!/bin/bash
P="fork_rollout.py"
pids=$(pgrep -f "$P"); echo "killing $pids"; [ -n "$pids" ] && kill -9 $pids
sleep 5; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
