#!/bin/bash
# login-node waiter: once global_step_200 of the 8B run is saved, stop the training (only its processes), then launch the evals
J=265672; CK=/localscratch/sya225/checkpoints/opd_q3i17_from8B_dapo_16k
until [ "$(srun --jobid=$J --overlap -c 1 --ntasks=1 cat $CK/latest_checkpointed_iteration.txt 2>/dev/null)" = "200" ]; do sleep 60; done
echo "step 200 saved $(date -Is)"; sleep 30
tmux kill-session -t q8b19 2>/dev/null
srun --jobid=$J --overlap -c 1 --ntasks=1 bash /home/sya225/slurm/kill_q8b.sh
echo "training stopped $(date -Is)"
tmux new -d -s evq8b 'srun --jobid=265672 --overlap -c 30 --ntasks=1 bash /home/sya225/opd/scripts/eval_q8b_ckpts.sh > /home/sya225/logs/evq8b.log 2>&1'
echo "eval launched $(date -Is)"
