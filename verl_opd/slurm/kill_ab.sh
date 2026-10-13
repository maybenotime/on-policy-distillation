#!/bin/bash
for pat in driver19ab.sh "teacher_lane2.sh 0 8350" "teacher_lane2.sh 1 8351" "teacher_lane2.sh 2 8352" "q3TP_A4B_shapeM500-s100" "q3TP_A4B_shapeM500_s100" "--port 835"; do pkill -f -- "$pat" 2>/dev/null; done
sleep 15; for pat in "q3TP_A4B_shapeM500_s100" "--port 835"; do pkill -9 -f -- "$pat" 2>/dev/null; done
sleep 5; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo; pgrep -fc "teacher_eval|vllm"
