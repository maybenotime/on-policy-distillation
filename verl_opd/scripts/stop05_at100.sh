#!/bin/bash
# 05: let Math-1200 16k reach global_step_100, then stop it. No eval.
source /home/sya225/opd/env.sh none
C=$SCRATCH/checkpoints/opd_code_17b_fromK4Bmath1200_16k
until [ "$(cat $C/latest_checkpointed_iteration.txt 2>/dev/null)" = "100" ]; do sleep 30; done
echo "step100 saved $(date -Is)"; sleep 90
P1="run_opd_code"; P2="main_ppo"; P3="ray::"; P4="vLLMHttpServer"
pkill -f "$P1"; sleep 5; pkill -f "$P2"; pkill -f "$P3"; pkill -f "$P4"; sleep 20; pkill -9 -f "$P3"; pkill -9 -f "$P2"
sleep 20; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
ls $C/global_step_100/actor | head -3
echo "######## STOPPED K4Bmath1200_16k at step 100 $(date -Is) ########"
