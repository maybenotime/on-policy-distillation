#!/bin/bash
# Run on the login node:  bash /home/sya225/slurm/kill05_hung.sh
# Kills the two hung 05 trainings (run A tr_qA, run C tr_qC; both stuck in actor_rollout_update_weights since 19:10/19:16 9-18).
# The out files are moved aside first so driver05e does not mistake the kill for a normal END.
SL=/home/sya225/slurm; T=$(date +%m%d_%H%M)
mv $SL/tr_qA.out $SL/tr_qA_hung_$T.out 2>/dev/null; mv $SL/tr_qC.out $SL/tr_qC_hung_$T.out 2>/dev/null
tmux kill-session -t tr_qA 2>/dev/null && echo "killed tr_qA"; tmux kill-session -t tr_qC 2>/dev/null && echo "killed tr_qC"
sleep 20
srun --jobid=269604 --overlap -c 1 --ntasks=1 bash -lc 'pkill -u sya225 -f "ray_qA|ray_qC|main_ppo" 2>/dev/null; sleep 10; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo' 2>/dev/null
echo "KILL05_DONE $(date -Is)"
