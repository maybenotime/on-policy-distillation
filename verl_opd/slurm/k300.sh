#!/bin/bash
P1="teacher_code_eval.sh K4Bcode300"; P2="tag K4Bcode300"; P3="fromRLCR4B\] DONE"
pkill -f "$P1"; pkill -f "$P2"; pkill -f "$P3"
sleep 3
ps -u sya225 -o pid,cmd | grep -E "tce|gen2|teacher_code" | grep -v grep | cut -c1-120
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
