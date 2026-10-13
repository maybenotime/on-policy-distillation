L=/localscratch/sya225/logs/train_17b_fromK4Bcode300_16k.log
date +%H:%M; grep -aiE "load.*checkpoint|resum.*(step|from)|Traceback" $L | tail -4 | cut -c1-200; grep -aoE " step:[0-9]+|timing_s/step:[0-9.]+" $L | tail -4; tail -n 2 $L | cut -c1-200; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
