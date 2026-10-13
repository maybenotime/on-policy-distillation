#!/bin/bash
# cs-venus-05 (job 269604), 9-19 08:55.  DeepSeek-series transplant on GPUs 3,4,5 (freed by stopping run A's 16k eval).
#   student DS-R1-Distill-1.5B;  A = R1-Distill-7B (ineffective, +0.0, pre-ρ 0.930);  B = JustRL-1.5B (effective, +9.1, pre-ρ 0.835)
# 1) 2-step smoke (batch 8, 512 tok) to check memory with a 7B HF teacher + a 1.5B one on each student rank
# 2) forward transplant ds15b_tpR17B_shapeJustRL, 200 steps, ckpt every 50
J=269604; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
A=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; B=hbx/JustRL-DeepSeek-1.5B
free345 () { srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tail -3 | awk "{s+=\$2} END {exit !(s<3000)}"' 2>/dev/null; }
echo "DRIVER05I_START $(date -Is)"
until free345; do sleep 60; done; echo "GPUS_345_FREE $(date -Is)"; sleep 30
srun --jobid=$J --overlap -c 1 --ntasks=1 bash -lc "rm -rf /localscratch/sya225/checkpoints/opd_ds15b_tp_smoke /tmp/ray_ds_smoke /localscratch/sya225/logs/train_ds15b_tp_smoke.log" 2>/dev/null
srun --jobid=$J --overlap -c 30 --ntasks=1 bash -lc "SMOKE=1 SAVE_FREQ=-1 bash $S/chain05_tp.sh ds15b_tp_smoke shape $A $B 3,4,5 /tmp/ray_ds_smoke" > $SL/smoke_ds_tp.out 2>&1
grep -q "END ds15b_tp_smoke rc=0" $SL/smoke_ds_tp.out || { echo "SMOKE FAILED $(date -Is)"; tail -3 $SL/smoke_ds_tp.out; exit 1; }
echo "SMOKE_DS_OK $(date -Is)"; sleep 30
tmux new-session -d -s tr_dsTP "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain05_tp.sh ds15b_tpR17B_shapeJustRL shape $A $B 3,4,5 /tmp/ray_dsTP 1.0 200 > $SL/tr_dsTP.out 2>&1"
echo "DS_TRANSPLANT_LAUNCHED $(date -Is)"
