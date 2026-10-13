#!/bin/bash
# cs-venus-05 (job 269604), 9-18 07:55.  Run B ended 07:49; run A (step ~94, ~12 min/step) will not end before ~9-19 05:00, and
# driver05e only launches run C after A ends.  This driver launches C on GPUs 0,1,2 as soon as B's evals (GPU 2) are done.
# Same tmux session / out-file names as driver05e (tr_qC, tr_qC.out): driver05e's own `tmux new-session -s tr_qC` will then fail
# as a duplicate, and its `until ended tr_qC` + ev_qC logic keeps working on this run.
J=269604; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
C_TAG=q3i17_fromQ8Bgrpomath_dapo_16k; C_T=xiaoyuchen1/Qwen3-8B-GRPO-MATH
n_done () { cat $SL/$1.out 2>/dev/null | grep -c EVAL_DONE; }
echo "DRIVER05G_START $(date -Is)"
until [ "$(n_done ev_qB)" -ge 2 ]; do sleep 120; done; echo "EV_QB_DONE $(date -Is)"; sleep 60
tmux has-session -t tr_qC 2>/dev/null && { echo "tr_qC already exists, not launching"; exit 0; }
tmux new-session -d -s tr_qC "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_q3.sh $C_TAG $C_T 0,1,2 /tmp/ray_qC > $SL/tr_qC.out 2>&1"
echo "RUN_C_LAUNCHED on 0,1,2 $(date -Is)"
