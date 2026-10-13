#!/bin/bash
# login-node waiter: when job 269301 starts on cs-venus-19, verify the s200 checkpoints survived and
# launch the three post-OPD probes (one per GPU).
J=269301
until [ "$(squeue -j $J -h -o %T 2>/dev/null)" = "RUNNING" ]; do sleep 120; done
echo "JOB_RUNNING $J $(date -Is)"
sleep 30
echo "CKPT_CHECK: $(srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'ls -d /localscratch/sya225/checkpoints/hf_ds15b_from*_step200 2>&1 | tr "\n" " "')"
for N in JustRL DS7B Sky7B; do
  tmux new-session -d -s pd$N "srun --jobid=$J --overlap -c 12 --ntasks=1 bash /home/sya225/opd/scripts/post_ds_$N.sh > /home/sya225/slurm/post_ds_$N.out 2>&1"
  sleep 20
done
echo "LAUNCHED_ALL $(date -Is)"
