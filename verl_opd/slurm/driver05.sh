#!/bin/bash
# login-node driver for the three new-teacher runs on cs-venus-05 (job 269604).
J=269604; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
run () { tmux new-session -d -s $1 "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $2 > $SL/$1.out 2>&1"; }
ev  () { tmux new-session -d -s $1 "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval05.sh $2 $3 $4 $5 $6 > $SL/$1.out 2>&1"; }
ended () { grep -q "######## END" $SL/$1.out 2>/dev/null; }
echo "DRIVER_START $(date -Is)"
run tr_nemo $S/chain05_nemo.sh
# stagger: start the second Ray/verl run only after the first has logged its first training step (deadlock otherwise)
t0=$(date +%s); until srun --jobid=$J --overlap -c 1 --ntasks=1 grep -aq " step:1 - " /localscratch/sya225/logs/train_ds15b_fromNemoBroRL_dapo_16k.log 2>/dev/null || [ $(( $(date +%s) - t0 )) -gt 3600 ]; do sleep 60; done
echo "NEMO_STEP1 $(date -Is)"; sleep 30
run tr_light $S/chain05_light.sh
until ended tr_nemo; do sleep 300; done; echo "NEMO_ENDED $(date -Is)"; sleep 90
run tr_dscaler $S/chain05_dscaler.sh
until ended tr_light; do sleep 300; done; echo "LIGHT_ENDED $(date -Is)"; sleep 60
# evals on GPUs 3,4,5 (no training there any more): GPU3 nemo s200 -> light s50 ; GPU4 nemo s50 ; GPU5 light s200
tmux new-session -d -s ev_g3 "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05.sh ds15b_fromNemoBroRL_dapo_16k 200 3 8091 NemoBroRL; bash $SL/eval05.sh ds15b_fromLight7B_dapo_16k 50 3 8094 Light7B' > $SL/ev_g3.out 2>&1"
ev ev_g4 ds15b_fromNemoBroRL_dapo_16k 50 4 8092 NemoBroRL
ev ev_g5 ds15b_fromLight7B_dapo_16k 200 5 8093 Light7B
until ended tr_dscaler; do sleep 300; done; echo "DSCALER_ENDED $(date -Is)"; sleep 60
ev ev_g0 ds15b_fromDeepScaleR_dapo_16k 200 0 8095 DeepScaleR
ev ev_g1 ds15b_fromDeepScaleR_dapo_16k 50 1 8096 DeepScaleR
echo "DRIVER_ALL_LAUNCHED $(date -Is)"
