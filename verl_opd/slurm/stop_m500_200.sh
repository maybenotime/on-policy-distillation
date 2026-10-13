#!/bin/bash
# login-node stopper: once global_step_200 of the Math-500 run is saved, kill its training processes (the chain script then
# prints its END line, which triggers wait_sky.sh to launch the Skywork run).
J=265672; CK=/localscratch/sya225/checkpoints/opd_q3i17_fromK4Bmath500_dapo_16k
until [ "$(srun --jobid=$J --overlap -c 1 --ntasks=1 cat $CK/latest_checkpointed_iteration.txt 2>/dev/null)" = "200" ]; do sleep 60; done
echo "step 200 saved $(date -Is)"; sleep 45
srun --jobid=$J --overlap -c 1 --ntasks=1 bash /home/sya225/slurm/kill_q8b.sh
echo "math500 training stopped $(date -Is)"
