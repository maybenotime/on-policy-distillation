#!/bin/bash
# kill chain A + Math-1200 run; run as a file so pkill -f doesn't match this shell
T=K4Bmath1200
kill 2614107 2614101 2>/dev/null
pkill -TERM -f "ray_$T" ; pkill -TERM -f "opd_code_17b_from$T"
sleep 20
pkill -KILL -f "ray_$T"; pkill -KILL -f "opd_code_17b_from$T"
for p in $(nvidia-smi -i 3,4,5 --query-compute-apps=pid --format=csv,noheader); do kill -KILL $p 2>/dev/null; done
sleep 5
echo left:; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
ps -u sya225 -o pid,cmd | grep -E "chain_teachers|$T" | grep -v grep | cut -c1-100
echo 4b steps: $(grep -c step: /localscratch/sya225/logs/train_17b_from4b.log)
