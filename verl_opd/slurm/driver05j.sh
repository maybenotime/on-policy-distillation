#!/bin/bash
# cs-venus-05 (job 269604), 9-19 10:35.  DeepSeek transplant: stop at step 50, then evaluate s50 (16k, DS student).
J=269604; SL=/home/sya225/slurm; CK=/localscratch/sya225/checkpoints; DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
echo "DRIVER05J_START $(date -Is)"
until srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc "[ -s $CK/opd_ds15b_tpR17B_shapeJustRL/global_step_50/actor/model_world_size_2_rank_0.pt ]" 2>/dev/null; do sleep 180; done
sleep 150; echo "DS_S50_READY $(date -Is)"
tmux kill-session -t tr_dsTP 2>/dev/null && echo "stopped the DS transplant at step ~50"
sleep 45; srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'pkill -u sya225 -f ray_dsTP 2>/dev/null' 2>/dev/null; sleep 30
tmux new-session -d -s ev_dsTP "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval19.sh ds15b_tpR17B_shapeJustRL 50 3 8182 dsTP_R17B_shapeJustRL $DS 16384 > $SL/ev_dsTP.out 2>&1"
echo "DS_EVAL_LAUNCHED (s50, 16k, gpu 3) $(date -Is)"
