#!/bin/bash
# login-node waiter (grep loop only): when the 4B math run ends, replace chain B (step 264964.5) with chain B2
until grep -q "END math16k 4b" /home/sya225/logs/mathB.log; do sleep 60; done
echo "4b ended $(date -Is)"; scancel 264964.5; sleep 30
srun --jobid=264964 --overlap -c 1 --ntasks=1 bash -c 'rm -rf /localscratch/sya225/checkpoints/opd_math_17b_fromK4Bmath500_16k /tmp/ray_math16k_K4Bmath500; rm -f /localscratch/sya225/logs/train_math17b_fromK4Bmath500_16k.log; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "'
tmux new -d -s mathB2 'srun --jobid=264964 --overlap -c 40 --ntasks=1 bash /home/sya225/opd/scripts/chain05_mathB2.sh > /home/sya225/logs/mathB2.log 2>&1'
echo "chain B2 launched $(date -Is)"
