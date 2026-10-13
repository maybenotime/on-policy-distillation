date +%H:%M
for f in /localscratch/sya225/logs/train_17b_fromK4Bcode300_16k.log; do echo "== 19 code300_16k"; grep -aoE " step:[0-9]+|timing_s/step:[0-9.]+|response_length/mean:[0-9.]+" $f | tail -6 | paste - - -; grep -ac Traceback $f; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
