#!/bin/bash
# cs-venus-05 (job 269604): three Qwen3-1.7B <- {8B-DAPO-NT, 4B-Code-1200, 8B-GRPO-MATH} OPD runs after the DS runs and
# their evals release the GPUs.  Pre-probes on GPU 2 as soon as DeepScaleR ends.  Evals (8k) after each run.
J=269604; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
A_TAG=q3i17_fromQ8Bdapo_dapo_16k;     A_T=pb09204048/Qwen3-8B-DAPO-iter449-disable-thinking; A_N=Q8Bdapo
B_TAG=q3i17_fromK4Bcode1200_dapo_16k; B_T=Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200;      B_N=K4Bcode1200
C_TAG=q3i17_fromQ8Bgrpomath_dapo_16k; C_T=xiaoyuchen1/Qwen3-8B-GRPO-MATH;                       C_N=Q8Bgrpomath
n_done () { cat $SL/$1.out 2>/dev/null | grep -c EVAL_DONE; }
ended () { grep -q '######## END' $SL/$1.out 2>/dev/null; }
step1 () { srun --jobid=$J --overlap -c 1 --ntasks=1 grep -aq " step:1 - " /localscratch/sya225/logs/train_$1.log 2>/dev/null; }
echo "DRIVER05C_START $(date -Is)"
# downloads (no GPU)
srun --jobid=$J --overlap -c 4 --ntasks=1 bash -lc "source /home/sya225/opd/env.sh none; \$CODE_EVAL_ENV/bin/python -c \"
from huggingface_hub import snapshot_download
for m in ['$A_T','$B_T','$C_T']: print('DL', snapshot_download(m, allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.py','*.jinja']))\"" 2>&1 | grep DL
# pre-probes on GPU 2 once DeepScaleR has ended
until grep -q DSCALER_ENDED $SL/driver05.out 2>/dev/null; do sleep 300; done
tmux new-session -d -s preq3 "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/pre05_q3three.sh 2 > $SL/pre05_q3three.out 2>&1"
# run A on GPUs 3,4,5 once the Nemotron/Light evals are done
until grep -q LIGHT_ENDED $SL/driver05.out 2>/dev/null && [ "$(n_done ev_g3)" -ge 2 ] && [ "$(n_done ev_g4)" -ge 1 ] && [ "$(n_done ev_g5)" -ge 1 ]; do sleep 300; done
echo "GPUS_345_FREE $(date -Is)"; sleep 60
tmux new-session -d -s tr_qA "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_q3.sh $A_TAG $A_T 3,4,5 /tmp/ray_qA > $SL/tr_qA.out 2>&1"
# run B on GPUs 0,1,2 once DeepScaleR's evals are done (and A has logged step 1: Ray stagger)
until grep -q DSCALER_ENDED $SL/driver05.out 2>/dev/null && [ "$(n_done ev_g0)" -ge 1 ] && [ "$(n_done ev_g1)" -ge 1 ] && grep -q PREQ3_DONE $SL/pre05_q3three.out 2>/dev/null; do sleep 300; done
t0=$(date +%s); until step1 $A_TAG || ended tr_qA || [ $(( $(date +%s) - t0 )) -gt 3600 ]; do sleep 60; done
echo "GPUS_012_FREE $(date -Is)"; sleep 60
tmux new-session -d -s tr_qB "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_q3.sh $B_TAG $B_T 0,1,2 /tmp/ray_qB > $SL/tr_qB.out 2>&1"
# run C on whichever set frees first; evals of A/B right after each ends (on one GPU of their set)
until ended tr_qA || ended tr_qB; do sleep 300; done
if ended tr_qA; then CG=3,4,5; CR=/tmp/ray_qC; EV_A=4; echo "A ENDED first $(date -Is)"; else CG=0,1,2; CR=/tmp/ray_qC; EV_B=1; echo "B ENDED first $(date -Is)"; fi
sleep 60
ended tr_qA && tmux new-session -d -s ev_qA "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05q.sh $A_TAG 200 5 8121 $A_N; bash $SL/eval05q.sh $A_TAG 50 5 8122 $A_N' > $SL/ev_qA.out 2>&1"
ended tr_qB && tmux new-session -d -s ev_qB "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05q.sh $B_TAG 200 2 8123 $B_N; bash $SL/eval05q.sh $B_TAG 50 2 8124 $B_N' > $SL/ev_qB.out 2>&1"
# C uses the two remaining GPUs of the freed set + the third is busy with the eval -> give C the freed set's first two GPUs + one from the other set only when both A and B ended
until ended tr_qA && ended tr_qB; do sleep 300; done; echo "A and B ENDED $(date -Is)"; sleep 60
[ -z "${EV_A:-}" ] && tmux new-session -d -s ev_qA "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05q.sh $A_TAG 200 5 8121 $A_N; bash $SL/eval05q.sh $A_TAG 50 5 8122 $A_N' > $SL/ev_qA.out 2>&1"
[ -z "${EV_B:-}" ] && tmux new-session -d -s ev_qB "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05q.sh $B_TAG 200 2 8123 $B_N; bash $SL/eval05q.sh $B_TAG 50 2 8124 $B_N' > $SL/ev_qB.out 2>&1"
sleep 30
tmux new-session -d -s tr_qC "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_q3.sh $C_TAG $C_T 0,1,3 /tmp/ray_qC > $SL/tr_qC.out 2>&1"
until ended tr_qC; do sleep 300; done; echo "C ENDED $(date -Is)"; sleep 60
tmux new-session -d -s ev_qC "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05q.sh $C_TAG 200 4 8125 $C_N; bash $SL/eval05q.sh $C_TAG 50 4 8126 $C_N' > $SL/ev_qC.out 2>&1"
echo "DRIVER05C_ALL_LAUNCHED $(date -Is)"
