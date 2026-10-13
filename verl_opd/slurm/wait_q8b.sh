#!/bin/bash
# login-node waiter: after the DS-7B training is stopped at step 200, launch Qwen3-1.7B <- Qwen3-8B (non-thinking, k1+PG) on GPUs 1-3
until grep -q "training stopped" /home/sya225/logs/stop_ds7b_200.log 2>/dev/null; do sleep 120; done
echo "ds7b stopped $(date -Is)"; sleep 120
tmux new -d -s q8b19 'srun --jobid=265672 --overlap -c 48 --ntasks=1 bash /home/sya225/opd/scripts/chain19_q3i8b.sh > /home/sya225/logs/q8b19.log 2>&1'
echo "q3i8b run launched $(date -Is)"
