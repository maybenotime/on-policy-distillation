#!/bin/bash
PAT='K4Bcode1200'
pgrep -u sya225 -f $PAT | grep -v $$ | xargs -r kill 2>/dev/null
sleep 5
pgrep -u sya225 -f $PAT | grep -v $$ | xargs -r kill -9 2>/dev/null
sleep 3
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "c1200 left: $(pgrep -u sya225 -f $PAT | wc -l)"
echo "4b alive: $(pgrep -u sya225 -f from4b | wc -l)"
tail -c 300 /localscratch/sya225/logs/train_17b_from4b.log | grep -oE "step:[0-9]+" | tail -1
