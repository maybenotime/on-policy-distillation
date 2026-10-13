#!/bin/bash
# login-node waiter: when the two parallel k3 evals on GPUs 0/1 finish, launch DS-1.5B <- DS-7B OPD on GPUs 0-2
until grep -q "EVAL_K3_REST_DONE" /home/sya225/logs/evk3r.log; do sleep 120; done
echo "evals on GPU0/1 done $(date -Is)"; sleep 60
tmux new -d -s ds7b19 'srun --jobid=265672 --overlap -c 48 --ntasks=1 bash /home/sya225/opd/scripts/chain19_ds7b.sh > /home/sya225/logs/ds7b19.log 2>&1'
echo "ds7b run launched $(date -Is)"
