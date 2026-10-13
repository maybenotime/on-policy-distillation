#!/bin/bash
# login-node waiter: (1) after the Skywork download, run the offline pre-probe on GPU 0; (2) after the Math-500 run ends,
# launch DS-R1-Distill-1.5B <- Skywork-OR1-Math-7B on GPUs 2,3,1 (teacher on GPU 1).
until grep -q "DL19B_DONE" /home/sya225/logs/dlsky.log 2>/dev/null; do sleep 60; done
echo "download done $(date -Is)"
tmux new -d -s presky 'srun --jobid=265672 --overlap -c 8 --ntasks=1 bash /home/sya225/opd/scripts/pre_dssky.sh > /home/sya225/logs/presky.log 2>&1'
until grep -q "######## END q3i17_fromK4Bmath500_dapo_16k" /home/sya225/logs/m50019.log 2>/dev/null; do sleep 120; done
echo "math500 run ended $(date -Is)"; sleep 120
tmux new -d -s sky19 'srun --jobid=265672 --overlap -c 48 --ntasks=1 bash /home/sya225/opd/scripts/chain19_dssky.sh > /home/sya225/logs/sky19.log 2>&1'
echo "skywork run launched $(date -Is)"
