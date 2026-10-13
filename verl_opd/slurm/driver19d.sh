#!/bin/bash
# cs-venus-19 (job 269301), 9-18 20:25.  s50 evals cancelled; the four remaining s100 evals run in PARALLEL on the now-idle
# training GPUs 0,1,2 instead of queueing behind GPU 3.  GPU 3 keeps the in-flight p2c_M500 s100 (started 18:16).
J=269301; SL=/home/sya225/slurm; E=$SL/eval19.sh
echo "DRIVER19D_START $(date -Is)"
ev () { tmux new-session -d -s $1 "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $E $2 100 $3 $4 $5 $6 $7 > $SL/$1.out 2>&1"; echo "launched $1 ($5 s100) on gpu $3 $(date -Is)"; }
DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; Q3=Qwen/Qwen3-1.7B
ev e19_rndJustRL ds15b_fromJustRL_dmRND 0 8140 rnd_JustRL $DS 16384   # DS 16k, longest
ev e19_p2wQ8B    q3i17_from8B_dmP2W     1 8142 p2w_Q8B    $Q3 8192
ev e19_p2cQ8B    q3i17_from8B_dmP2C     2 8144 p2c_Q8B    $Q3 8192
until grep -q EVAL_DONE $SL/e19_p2wQ8B.out 2>/dev/null; do sleep 120; done; echo "p2w_Q8B done $(date -Is)"; sleep 30
ev e19_rndM500   q3i17_fromM500_dmRND   1 8146 rnd_M500   $Q3 8192
until grep -q EVAL_DONE $SL/e19_rndM500.out 2>/dev/null && grep -q EVAL_DONE $SL/e19_p2cQ8B.out 2>/dev/null && grep -q EVAL_DONE $SL/e19_rndJustRL.out 2>/dev/null; do sleep 300; done
echo "DRIVER19D_ALL_DONE $(date -Is)"
for f in e19_rndJustRL e19_p2wQ8B e19_p2cQ8B e19_rndM500; do grep -h '^rnd\|^p2' $SL/$f.out | tail -1; done
