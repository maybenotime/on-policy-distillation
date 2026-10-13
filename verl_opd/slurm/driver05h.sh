#!/bin/bash
# cs-venus-05 (job 269604), 9-18 20:10.  After the user runs kill05_hung.sh: wait for GPUs 0-5 to be free, then resume
# run A (3,4,5; verl resume_mode=auto -> global_step_150) and run C (0,1,2; -> global_step_50), Ray-staggered.
J=269604; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
A_TAG=q3i17_fromQ8Bdapo_dapo_16k; A_T=pb09204048/Qwen3-8B-DAPO-iter449-disable-thinking
C_TAG=q3i17_fromQ8Bgrpomath_dapo_16k; C_T=xiaoyuchen1/Qwen3-8B-GRPO-MATH
step1 () { srun --jobid=$J --overlap -c 1 --ntasks=1 grep -aq " step:15[1-9] - \| step:1[6-9][0-9] - " /localscratch/sya225/logs/train_$1.log 2>/dev/null; }
gpus_free () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {exit !(s<2000)}"' 2>/dev/null; }
echo "DRIVER05H_START $(date -Is)"
until [ -f $SL/kill05_hung.done ] || ! tmux has-session -t tr_qA 2>/dev/null; do sleep 60; done
until gpus_free; do sleep 60; done; echo "GPUS_FREE $(date -Is)"; sleep 30
tmux new-session -d -s tr_qA "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_q3.sh $A_TAG $A_T 3,4,5 /tmp/ray_qA > $SL/tr_qA.out 2>&1"
echo "RUN_A_RESUMED $(date -Is)"
t0=$(date +%s); until step1 $A_TAG || [ $(( $(date +%s) - t0 )) -gt 2400 ]; do sleep 60; done
tmux new-session -d -s tr_qC "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_q3.sh $C_TAG $C_T 0,1,2 /tmp/ray_qC > $SL/tr_qC.out 2>&1"
echo "RUN_C_RESUMED $(date -Is)"
