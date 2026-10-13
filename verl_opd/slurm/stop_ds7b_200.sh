#!/bin/bash
# login-node waiter: once global_step_200 of the DS-7B run is saved, stop the training step, then launch the MathArena eval
# of step 200 (16k, 7 comps, same setting as the JustRL s200 eval) on GPU 0.
J=265672; CK=/localscratch/sya225/checkpoints/opd_ds15b_fromDS7B_dapo_16k
until [ "$(srun --jobid=$J --overlap -c 1 --ntasks=1 cat $CK/latest_checkpointed_iteration.txt 2>/dev/null)" = "200" ]; do sleep 120; done
echo "step 200 saved $(date -Is)"; sleep 30
scancel $(squeue -s -j $J -h -o '%i' | grep -vE 'batch|extern' | tr '\n' ' '); sleep 20; tmux kill-session -t ds7b19 2>/dev/null
srun --jobid=$J --overlap -c 1 --ntasks=1 bash -c 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo'
echo "training stopped $(date -Is)"
tmux new -d -s evds7b 'srun --jobid=265672 --overlap -c 16 --ntasks=1 bash /home/sya225/opd/scripts/eval_ds7b_s200.sh > /home/sya225/logs/evds7b.log 2>&1'
echo "eval launched $(date -Is)"
