#!/bin/bash
# cs-venus-05 (job 269604), 9-19 10:45.  "shape-only" cell of the knowledge x shape 2x2 on the DeepSeek pair:
#   student DS-R1-Distill-1.5B;  preferences = the student's own (base) distribution;  shape = JustRL-1.5B.
#   -> ds15b_tpStu_shapeJustRL, 50 steps, then a 16k s50 eval.
# Also owns what driver05e can no longer do correctly (its ev_qC would land on GPU 4, which the DS transplant holds):
#   run C's s200 8k eval goes to GPU 5 once the DS transplant has been stopped at step 50 (driver05j).
J=269604; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; B=hbx/JustRL-DeepSeek-1.5B; Q3=Qwen/Qwen3-1.7B
freegpu () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v g=$1 '\$1==g {exit !(\$2<3000)}'" 2>/dev/null; }
free012 () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -3 | awk "{s+=\$2} END {exit !(s<3000)}"' 2>/dev/null; }
echo "DRIVER05K_START $(date -Is)"
until grep -q '######## END' $SL/tr_qC.out 2>/dev/null; do sleep 120; done; echo "run C ENDED $(date -Is)"
until free012; do sleep 60; done; sleep 30
tmux new-session -d -s tr_dsStu "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_tp.sh ds15b_tpStu_shapeJustRL shape $DS $B 0,1,2 /tmp/ray_dsStu 1.0 50 > $SL/tr_dsStu.out 2>&1"
echo "SHAPE_ONLY_RUN_LAUNCHED (50 steps, gpus 0,1,2) $(date -Is)"
# run C's evals on GPU 5 once the DS transplant releases it
until freegpu 5; do sleep 180; done; sleep 30
tmux new-session -d -s ev_qC "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval05q.sh q3i17_fromQ8Bgrpomath_dapo_16k 200 5 8185 Q8Bgrpomath > $SL/ev_qC.out 2>&1"
echo "EV_QC_LAUNCHED gpu 5 $(date -Is)"
# the shape-only run's own eval (16k, DS student) when it finishes
until grep -q '######## END' $SL/tr_dsStu.out 2>/dev/null; do sleep 180; done; echo "shape-only run ENDED $(date -Is)"; sleep 60
tmux new-session -d -s ev_dsStu "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval19.sh ds15b_tpStu_shapeJustRL 50 4 8187 dsTP_Stu_shapeJustRL $DS 16384 > $SL/ev_dsStu.out 2>&1"
echo "EV_SHAPE_ONLY_LAUNCHED gpu 4 $(date -Is)"
