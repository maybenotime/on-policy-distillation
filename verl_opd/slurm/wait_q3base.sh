#!/bin/bash
# login-node waiter (grep loop only): when the DeepSeek run ends, start the Qwen3-1.7B-Base <- 4B-Base-GRPO run
until grep -q "######## END ds15b_fromJustRL_dapo_16k" /home/sya225/logs/ds19.log; do sleep 120; done
echo "ds run ended $(date -Is)"; sleep 90
tmux new -d -s q3b19 'srun --jobid=265672 --overlap -c 48 --ntasks=1 bash /home/sya225/opd/scripts/chain19_q3base.sh > /home/sya225/logs/q3b19.log 2>&1'
echo "q3base launched $(date -Is)"
