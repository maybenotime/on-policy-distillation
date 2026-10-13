#!/bin/bash
# 8B-DAPO's 8k eval is in the serial re-prompt tail (ETA 5 h).  Split like Skywork: resume the main tag on GPU 1 restricted to
# the AIME/HMMT-25 comps, siblings for beyondaime (GPU 0) and hmmt26+amc (GPU 2).  Merge with ma_merge.py.
H=/home/sya225/logs; T=/home/sya225/slurm/teacher_lane2.sh
echo "DRIVER05Z_START $(date -Is)"
for pat in "teacher_lane.sh 1 8311" "teacher_eval.sh tQ8BDAPO-8k" "tQ8BDAPO_8k" "--port 8311"; do pkill -f -- "$pat" 2>/dev/null; done
sleep 20; for pat in "tQ8BDAPO_8k" "--port 8311"; do pkill -9 -f -- "$pat" 2>/dev/null; done
for i in $(seq 1 40); do m=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 3000 ] && break; sleep 15; done
echo "GPU1_FREE mem=$m $(date -Is)"
M=pb09204048/Qwen3-8B-DAPO-iter449-disable-thinking
COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025" bash $T 1 8311 tQ8BDAPO-8k:$M:8192 > $H/tl05_1b.out 2>&1 &
COMPS="custom/beyondaime"          bash $T 0 8312 tQ8BDAPO-8kb:$M:8192 > $H/tl05_0b.out 2>&1 &
COMPS="hmmt/hmmt_feb_2026 custom/amc" bash $T 2 8313 tQ8BDAPO-8kc:$M:8192 > $H/tl05_2b.out 2>&1 &
wait
echo "DRIVER05Z_DONE $(date -Is)"
