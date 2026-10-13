#!/bin/bash
tmux kill-session -t chainB3 2>/dev/null
tmux kill-session -t eval4b 2>/dev/null
srun --jobid=263392 --overlap -c 1 --ntasks=1 bash -c '
pkill -u sya225 -f "chainB3|chain_teachers" ; sleep 1
for p in $(pgrep -u sya225 -f "ray_K4Bcode1200|fromK4Bcode1200"); do kill $p 2>/dev/null; done
sleep 5
pkill -9 -u sya225 -f "ray_K4Bcode1200|fromK4Bcode1200"
sleep 3
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
pgrep -u sya225 -af "main_ppo|ray::|VLLM" | grep -c . '
