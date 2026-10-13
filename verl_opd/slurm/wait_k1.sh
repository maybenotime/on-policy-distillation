#!/bin/bash
# login-node waiter: when the k3 Base run ends, start the same pair with k1 + PG (k1_probe), fresh
until grep -q "######## END q3b17_fromQ3B4Bgrpo_dapo_16k_k3" /home/sya225/logs/q3bk3.log; do sleep 120; done
echo "k3 run ended $(date -Is)"; sleep 90
tmux new -d -s q3bk1 'srun --jobid=265672 --overlap -c 48 --ntasks=1 bash /home/sya225/opd/scripts/chain19_q3base.sh > /home/sya225/logs/q3bk1.log 2>&1'
echo "k1 run launched $(date -Is)"
