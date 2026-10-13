#!/bin/bash
# cs-venus-19 (job 270249), 9-19 08:40.  Replaces driver19f after the user stopped run 5 early (plateau reached by step 30).
# Sequence once the GPUs are free:
#   GPU 3 : eval run5 @ s50 (its only checkpoint), then run3 @ s50 (matched comparison point for the control)
#   0,1,2 : run 4  tpM500_shapeA4B   (reverse transplant: B's preferences + A's shape)  -> eval s200
#           run 6  tpA4B_shapeStu    (A's preferences + the student's own shape)        -> eval s200
J=270249; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
A=Qwen/Qwen3-4B; B=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500; Q3=Qwen/Qwen3-1.7B
ended () { grep -q '######## END' $SL/$1.out 2>/dev/null; }
free012 () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -3 | awk "{s+=\$2} END {exit !(s<3000)}"' 2>/dev/null; }
evl () { tmux new-session -d -s ev_$1 "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval19.sh q3i17_$2 $3 3 $4 $1 $Q3 8192 > $SL/ev_$1.out 2>&1"; echo "eval $1 (s$3) launched on gpu 3 $(date -Is)"; }
trn () { echo "$1 START $(date -Is)"
  tmux new-session -d -s tr_$1 "srun --jobid=$J --overlap -c 40 --ntasks=1 bash $S/chain19_tp.sh q3i17_$1 $2 $3 $4 0,1,2 /tmp/ray_$1 1.0 200 > $SL/tr_$1.out 2>&1"
  until ended tr_$1; do sleep 300; done; echo "$1 ENDED $(date -Is)"; sleep 60; }
echo "DRIVER19G_START $(date -Is)"
until free012; do sleep 60; done; echo "GPUS_FREE $(date -Is)"; sleep 30
evl tpA4B_uniform    tpA4B_uniform    50 8170          # the control, at its plateau
trn tpM500_shapeA4B  shape $B $A                       # reverse transplant
evl tpM500_shapeA4B  tpM500_shapeA4B  200 8172
trn tpA4B_shapeStu   student $A -                      # A's preferences + student's own shape
evl tpA4B_shapeStu   tpA4B_shapeStu   200 8174
echo "DRIVER19G_ALL_LAUNCHED $(date -Is)"
