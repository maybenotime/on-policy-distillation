#!/bin/bash
# cs-venus-05 (job 269604), 9-19 15:40.  run C's s200 eval failed at 10:49: vLLM asked for 0.9 of GPU 5 while the DS
# transplant still held 40 GB there, and MathArena then retried the dead port for 5 h.  Relaunch it once a GPU is really
# free, with GPU_UTIL=0.45 so it cannot happen again.
J=269604; SL=/home/sya225/slurm
freegpu () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v g=$1 '\$1==g {exit !(\$2<3000)}'" 2>/dev/null; }
echo "DRIVER05L_START $(date -Is)"
until freegpu 5; do sleep 120; done; sleep 30
tmux new-session -d -s ev_qC2 "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -lc 'GPU_UTIL=0.45 bash $SL/eval05q.sh q3i17_fromQ8Bgrpomath_dapo_16k 200 5 8195 Q8Bgrpomath' > $SL/ev_qC2.out 2>&1"
echo "EV_QC_RELAUNCHED gpu 5 util 0.45 $(date -Is)"
