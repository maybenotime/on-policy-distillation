#!/bin/bash
# cs-venus-19 (NEW job 270249, 6 d, started 9-18 20:33).  s100-only evals, 4 in parallel on GPUs 0-3.
# p2c_M500 was interrupted mid-eval by the job swap; MathArena resumes from its saved per-problem runs.
J=270249; SL=/home/sya225/slurm; E=$SL/eval19.sh
echo "DRIVER19E_START $(date -Is) job=$J"
ev () { tmux new-session -d -s $1 "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $E $2 100 $3 $4 $5 $6 $7 > $SL/$1.out 2>&1"; echo "launched $1 ($5 s100, ${7}tok) on gpu $3 $(date -Is)"; }
DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; Q3=Qwen/Qwen3-1.7B
ev e19_rndJustRL ds15b_fromJustRL_dmRND 0 8150 rnd_JustRL $DS 16384   # DS 16k, longest
ev e19_p2wQ8B    q3i17_from8B_dmP2W     1 8152 p2w_Q8B    $Q3 8192
ev e19_p2cQ8B    q3i17_from8B_dmP2C     2 8154 p2c_Q8B    $Q3 8192
ev e19_p2cM500   q3i17_fromM500_dmP2C   3 8156 p2c_M500   $Q3 8192
until grep -q EVAL_DONE $SL/e19_p2wQ8B.out 2>/dev/null; do sleep 120; done; echo "p2w_Q8B done $(date -Is)"; sleep 30
ev e19_rndM500   q3i17_fromM500_dmRND   1 8158 rnd_M500   $Q3 8192
until grep -q EVAL_DONE $SL/e19_rndM500.out 2>/dev/null && grep -q EVAL_DONE $SL/e19_p2cQ8B.out 2>/dev/null \
   && grep -q EVAL_DONE $SL/e19_p2cM500.out 2>/dev/null && grep -q EVAL_DONE $SL/e19_rndJustRL.out 2>/dev/null; do sleep 300; done
echo "DRIVER19E_ALL_DONE $(date -Is)"
for f in e19_rndJustRL e19_p2wQ8B e19_p2cQ8B e19_p2cM500 e19_rndM500; do tail -1 $SL/$f.out; done
