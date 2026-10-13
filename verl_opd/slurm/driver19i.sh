#!/bin/bash
# cs-venus-19 (job 270249), 9-19 12:50.  "knowledge-only on the EFFECTIVE teacher":
#   q3i17_tpM500_shapeStu = Math-Step500's preferences + the student's OWN per-position entropy (mode=student).
#   The shape gap is zero by construction (ρ(q̃, student) = 1, dH = 0), so whatever gain survives comes from preferences alone.
# Runs on GPUs 0,1,2 after run 6 finishes; 50 steps; s50 eval on GPU 3 once the other evals there are done.
J=270249; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
M500=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500; Q3=Qwen/Qwen3-1.7B
free012 () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -3 | awk "{s+=\$2} END {exit !(s<3000)}"' 2>/dev/null; }
free3 () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk "\$1==3 {exit !(\$2<3000)}"' 2>/dev/null; }
echo "DRIVER19I_START $(date -Is)"
until grep -q '######## END' $SL/tr_tpA4B_shapeStu.out 2>/dev/null; do sleep 120; done; echo "run 6 ended $(date -Is)"
until free012; do sleep 60; done; sleep 30
tmux new-session -d -s tr_tpM500_shapeStu "srun --jobid=$J --overlap -c 40 --ntasks=1 bash $S/chain19_tp.sh q3i17_tpM500_shapeStu student $M500 - 0,1,2 /tmp/ray_M500stu 1.0 50 > $SL/tr_tpM500_shapeStu.out 2>&1"
echo "M500_shapeStu START (50 steps) $(date -Is)"
until grep -q '######## END' $SL/tr_tpM500_shapeStu.out 2>/dev/null; do sleep 180; done; echo "M500_shapeStu ENDED $(date -Is)"; sleep 60
until free3; do sleep 180; done; sleep 30
tmux new-session -d -s ev_tpM500_shapeStu "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval19.sh q3i17_tpM500_shapeStu 50 3 8190 tpM500_shapeStu $Q3 8192 > $SL/ev_tpM500_shapeStu.out 2>&1"
echo "EVAL_LAUNCHED $(date -Is)"
