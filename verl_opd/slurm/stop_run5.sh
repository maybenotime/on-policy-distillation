#!/bin/bash
# Run on the login node:  bash /home/sya225/slurm/stop_run5.sh
# Stops run 5 (uniform control) at step ~75 -- its probes plateaued by step 30 -- and the driver19f chain that owns it.
# driver19g (already waiting) then evaluates run 5 @ s50 on GPU 3 and starts the reverse transplant on GPUs 0,1,2.
SL=/home/sya225/slurm
tmux kill-session -t driver19f2 2>/dev/null && echo "killed driver19f2"
tmux kill-session -t driver19f  2>/dev/null && echo "killed driver19f"
tmux kill-session -t tr_tpA4B_uniform 2>/dev/null && echo "killed tr_tpA4B_uniform"
sleep 20
srun --jobid=270249 --overlap -c 1 --ntasks=1 bash -lc 'pkill -u sya225 -f "ray_tpA4B_uniform" 2>/dev/null; sleep 8; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo' 2>/dev/null
echo "STOP_RUN5_DONE $(date -Is)"
