#!/bin/bash
# cs-venus-19 (job 270249), 9-19 10:35.  User: all transplant runs only need 50 steps, then evaluate.
#   GPU 3 : eval run3 @ s50 now (matched point for the controls), then run4 @ s50, then run6 @ s50
#   0,1,2 : run 4 is stopped as soon as its global_step_50 is on disk; run 6 (A ⊕ shape(student)) then runs 50 steps
J=270249; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; CK=/localscratch/sya225/checkpoints
A=Qwen/Qwen3-4B; Q3=Qwen/Qwen3-1.7B
have50 () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc "[ -s $CK/opd_q3i17_$1/global_step_50/actor/model_world_size_2_rank_0.pt ]" 2>/dev/null; }
free012 () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -3 | awk "{s+=\$2} END {exit !(s<3000)}"' 2>/dev/null; }
evdone () { grep -q EVAL_DONE $SL/ev_$1.out 2>/dev/null; }
evl () { tmux new-session -d -s ev_$1 "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval19.sh q3i17_$1 50 3 $2 $1 $Q3 8192 > $SL/ev_$1.out 2>&1"; echo "eval $1 s50 on gpu 3 $(date -Is)"; }
echo "DRIVER19H_START $(date -Is)"
evl tpA4B_shapeM500 8176                                   # run 3 at the matched checkpoint
until have50 tpM500_shapeA4B; do sleep 120; done; sleep 150   # let the write finish
echo "RUN4_S50_READY $(date -Is)"
tmux kill-session -t tr_tpM500_shapeA4B 2>/dev/null && echo "stopped run 4 at step ~50"
sleep 45; srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'pkill -u sya225 -f ray_tpM500_shapeA4B 2>/dev/null' 2>/dev/null
until free012; do sleep 60; done; sleep 30
tmux new-session -d -s tr_tpA4B_shapeStu "srun --jobid=$J --overlap -c 40 --ntasks=1 bash $S/chain19_tp.sh q3i17_tpA4B_shapeStu student $A - 0,1,2 /tmp/ray_shapeStu 1.0 50 > $SL/tr_tpA4B_shapeStu.out 2>&1"
echo "run 6 (50 steps) START $(date -Is)"
until evdone tpA4B_shapeM500; do sleep 120; done; sleep 30
evl tpM500_shapeA4B 8178
until grep -q '######## END' $SL/tr_tpA4B_shapeStu.out 2>/dev/null; do sleep 180; done; echo "run 6 ENDED $(date -Is)"
until evdone tpM500_shapeA4B; do sleep 120; done; sleep 30
evl tpA4B_shapeStu 8180
echo "DRIVER19H_ALL_LAUNCHED $(date -Is)"
