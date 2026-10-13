#!/bin/bash
# Stop the Code-300 16k OPD training on cs-venus-19 (chain19b). Resume later by
# re-running the same `run K4Bcode300_16k ...` line: verl resume_mode=auto picks
# up latest_checkpointed_iteration.txt in CKPT_DIR. Pattern kept in a variable so
# pkill -f does not match this script's own cmdline.
P1="run_opd_code"
P2="main_distillation"
P3="ray::"
pkill -f "$P1"; sleep 2
pkill -f "$P2"; pkill -f "$P3"
sleep 5
pkill -9 -f "$P2"; pkill -9 -f "$P3"
pkill -f "raylet"; pkill -f "gcs_server"; pkill -f "ray/dashboard"
sleep 3
echo "--- remaining python/ray procs of sya225:"
ps -u sya225 -o pid,etime,pcpu,cmd | grep -E "ray|verl|distill" | grep -v grep | cut -c1-140
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
cat /localscratch/sya225/checkpoints/opd_code_17b_fromK4Bcode300_16k/latest_checkpointed_iteration.txt
