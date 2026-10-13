L=/localscratch/sya225/logs/train_17b_fromK4Bmath1200_16k.log
date +%H:%M; grep -ac Traceback $L; grep -aoE " step:[0-9]+|response_length/mean:[0-9.]+|response_length/clip_ratio:[0-9.]+|timing_s/step:[0-9.]+" $L | tail -8 | paste - - - -; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
