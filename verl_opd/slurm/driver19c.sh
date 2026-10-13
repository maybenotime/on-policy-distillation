#!/bin/bash
# cs-venus-19 (job 269301): two random-data controls after driver19b's eight DeepMath-subset runs.
#   9  ds15b_fromJustRL_dmRND   DS-1.5B   <- JustRL-DeepSeek-1.5B      data verl_data/deepmath_rnd (uniform random 7000 of DeepMath-103K, seed 0)
#  10  q3i17_fromM500_dmRND     Qwen3-1.7B <- Qwen3-4B-Math-Step500     same data
# Sequential on GPUs 0,1,2 (chain_dm.sh, one epoch = 109 steps); evals appended to eval_queue19.txt (GPU-3 queue, s100 + s50).
J=269301; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; D=/home/sya225/opd/verl_data; Q=$SL/eval_queue19.txt
JUSTRL=hbx/JustRL-DeepSeek-1.5B; M500=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
ended () { grep -q '######## END' $SL/$1.out 2>/dev/null; }
run () { local TAG=$1 STU=$2 TEA=$3 DD=$4 NAME=$5 BASE=$6 MT=$7
  echo "$NAME START $(date -Is)"
  tmux new-session -d -s tr_$NAME "srun --jobid=$J --overlap -c 40 --ntasks=1 bash $S/chain_dm.sh $TAG $STU $TEA $DD 0,1,2 /tmp/ray_$NAME > $SL/tr_$NAME.out 2>&1"
  until ended tr_$NAME; do sleep 300; done; echo "$NAME ENDED $(date -Is)"; echo "$TAG $NAME $BASE $MT" >> $Q; sleep 60; }
echo "DRIVER19C_START $(date -Is)"
until grep -q DRIVER19B_ALL_TRAINED $SL/driver19b.out 2>/dev/null; do sleep 300; done; sleep 60
run ds15b_fromJustRL_dmRND deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B $JUSTRL $D/deepmath_rnd rnd_JustRL deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 16384
run q3i17_fromM500_dmRND   Qwen/Qwen3-1.7B $M500 $D/deepmath_rnd rnd_M500 Qwen/Qwen3-1.7B 8192
echo "DRIVER19C_ALL_TRAINED $(date -Is)"
