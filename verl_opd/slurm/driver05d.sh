#!/bin/bash
# cs-venus-05 (job 269604), written 9-17 07:55.  Recovery driver after the Nemo-s200 MathArena grader (PID 90877,
# 325 GB RSS, stuck since 05:04 grading hmmt_feb_2026 P31) got the DeepScaleR run OOM-killed at step 121.
# Coexists with driver05c: driver05c still launches run A on GPUs 3,4,5; its run-B gate is held by moving ev_g0.out
# aside so that the DeepScaleR resume (step 100 -> 200) can use GPUs 0,1,2 first.  This driver then runs the
# DeepScaleR s200 eval, run B, the A/B evals, run C and its eval.
J=269604; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
A_TAG=q3i17_fromQ8Bdapo_dapo_16k;     A_N=Q8Bdapo
B_TAG=q3i17_fromK4Bcode1200_dapo_16k; B_T=Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200; B_N=K4Bcode1200
C_TAG=q3i17_fromQ8Bgrpomath_dapo_16k; C_T=xiaoyuchen1/Qwen3-8B-GRPO-MATH;                  C_N=Q8Bgrpomath
n_done () { cat $SL/$1.out 2>/dev/null | grep -c EVAL_DONE; }
ended () { grep -q '######## END' $SL/$1.out 2>/dev/null; }
step1 () { srun --jobid=$J --overlap -c 1 --ntasks=1 grep -aq " step:1 - " /localscratch/sya225/logs/train_$1.log 2>/dev/null; }
alive () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc "ps -p 90877 -o comm= | grep -q python" 2>/dev/null; }
echo "DRIVER05D_START $(date -Is)"
# 1. hold driver05c's run-B gate (it counts EVAL_DONE lines in ev_g0.out)
[ -f $SL/ev_g0.out ] && mv $SL/ev_g0.out $SL/ev_g0_held.out && echo "EV_G0_HELD $(date -Is)"
# 2. Nemo s200 re-eval on GPU 4 once the stuck grader is gone (MathArena reuses the saved runs; only the missing ones are regenerated)
while alive; do sleep 60; done; echo "PID90877_GONE $(date -Is)"; sleep 30
tmux new-session -d -s ev_g4b "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval05.sh ds15b_fromNemoBroRL_dapo_16k 200 4 8097 NemoBroRL > $SL/ev_g4b.out 2>&1"
# 3. resume DeepScaleR (verl resume_mode=auto -> global_step_100) on 0,1,2 once ev_g0 (fails: no s200), ev_g1 (s50) and the Qwen3 pre-probes are done
until [ "$(n_done ev_g0_held)" -ge 1 ] && [ "$(n_done ev_g1)" -ge 1 ] && grep -q PREQ3_DONE $SL/pre05_q3three.out 2>/dev/null; do sleep 120; done
echo "GPUS_012_FREE_FOR_RESUME $(date -Is)"; sleep 60
tmux new-session -d -s tr_dscaler2 "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_dscaler.sh > $SL/tr_dscaler2.out 2>&1"
until ended tr_dscaler2; do sleep 300; done; echo "DSCALER2_ENDED $(date -Is)"; sleep 60
tmux new-session -d -s ev_g0b "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval05.sh ds15b_fromDeepScaleR_dapo_16k 200 0 8098 DeepScaleR > $SL/ev_g0b.out 2>&1"
# A's eval as soon as A ends (GPU 5), independent of the rest
( until ended tr_qA; do sleep 300; done; echo "A ENDED $(date -Is)"; sleep 60
  tmux new-session -d -s ev_qA "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05q.sh $A_TAG 200 5 8121 $A_N; bash $SL/eval05q.sh $A_TAG 50 5 8122 $A_N' > $SL/ev_qA.out 2>&1" ) &
# 4. run B on 0,1,2 after the DeepScaleR s200 eval (Ray stagger: A must have logged step 1, or 1 h)
until [ "$(n_done ev_g0b)" -ge 1 ]; do sleep 300; done
t0=$(date +%s); until step1 $A_TAG || ended tr_qA || [ $(( $(date +%s) - t0 )) -gt 3600 ]; do sleep 60; done
echo "GPUS_012_FREE $(date -Is)"; sleep 60
tmux new-session -d -s tr_qB "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_q3.sh $B_TAG $B_T 0,1,2 /tmp/ray_qB > $SL/tr_qB.out 2>&1"
until ended tr_qB; do sleep 300; done; echo "B ENDED $(date -Is)"; sleep 60
tmux new-session -d -s ev_qB "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05q.sh $B_TAG 200 2 8123 $B_N; bash $SL/eval05q.sh $B_TAG 50 2 8124 $B_N' > $SL/ev_qB.out 2>&1"
# 5. run C on 0,1,3 once A and B have both ended (ev_qA on 5, ev_qB on 2 keep running)
until ended tr_qA; do sleep 300; done; echo "A and B ENDED $(date -Is)"; sleep 90
tmux new-session -d -s tr_qC "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_q3.sh $C_TAG $C_T 0,1,3 /tmp/ray_qC > $SL/tr_qC.out 2>&1"
until ended tr_qC; do sleep 300; done; echo "C ENDED $(date -Is)"; sleep 60
tmux new-session -d -s ev_qC "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05q.sh $C_TAG 200 4 8125 $C_N; bash $SL/eval05q.sh $C_TAG 50 4 8126 $C_N' > $SL/ev_qC.out 2>&1"
wait; echo "DRIVER05D_ALL_LAUNCHED $(date -Is)"
