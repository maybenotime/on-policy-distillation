#!/bin/bash
# cs-venus-05 (job 269604), 9-17 12:40.  The Light-R1 s50 MathArena grader (PID 175482) hung the same way the Nemo-s200 one
# did (memory runaway while grading one response).  This driver: holds driver05c's run-A gate (ev_g3.out -> ev_g3_held.out),
# waits for the user to kill 175482, re-runs the Light s50 eval on GPU 3 (MathArena reuses the 191 saved problems; grader
# now patched with a length cap / timeout / 64 GB RLIMIT), then launches run A on 3,4,5 itself (same session/out names, so
# driver05e's A-eval / run-C logic keeps working).
J=269604; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
A_TAG=q3i17_fromQ8Bdapo_dapo_16k; A_T=pb09204048/Qwen3-8B-DAPO-iter449-disable-thinking
n_done () { cat $SL/$1.out 2>/dev/null | grep -c EVAL_DONE; }
alive () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc "ps -p 175482 -o comm= | grep -q python" 2>/dev/null; }
echo "DRIVER05F_START $(date -Is)"
[ -f $SL/ev_g3.out ] && mv $SL/ev_g3.out $SL/ev_g3_held.out && echo "EV_G3_HELD $(date -Is)"
while alive; do sleep 60; done; echo "PID175482_GONE $(date -Is)"; sleep 45
tmux new-session -d -s ev_g3b "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval05.sh ds15b_fromLight7B_dapo_16k 50 3 8099 Light7B > $SL/ev_g3b.out 2>&1"
until [ "$(n_done ev_g3b)" -ge 1 ]; do sleep 120; done; echo "LIGHT_S50_EVAL_DONE $(date -Is)"; sleep 60
tmux new-session -d -s tr_qA "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_q3.sh $A_TAG $A_T 3,4,5 /tmp/ray_qA > $SL/tr_qA.out 2>&1"
echo "RUN_A_LAUNCHED $(date -Is)"
