#!/bin/bash
# Skywork-OR1-7B's 16k eval is in MathArena's serial re-prompt tail (ETA 21 h, node 19 expires ~9-24 20:00).  Split it:
# kill the GPU-2 lane (Skywork + queued Q4BBaseGRPO), resume Skywork on GPU 2 restricted to the 4 AIME/HMMT-25 comps
# (per-problem status files are kept, so finished problems are not redone), and run the other comps under sibling tags on the
# idle GPUs 0 and 3; Q4BBaseGRPO follows on GPU 3.  Scores are merged with ma_merge.py.
H=/home/sya225/logs; T=/home/sya225/slurm/teacher_lane2.sh
echo "DRIVER19V_START $(date -Is)"
for pat in "teacher_lane.sh 2 8302" "teacher_eval.sh tSkywork7B-16k" "tSkywork7B_16k" "port 8302" "--port 8302"; do pkill -f -- "$pat" 2>/dev/null; done
sleep 20; for pat in "tSkywork7B_16k" "--port 8302"; do pkill -9 -f -- "$pat" 2>/dev/null; done
for i in $(seq 1 40); do m=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 3000 ] && break; sleep 15; done
echo "GPU2_FREE mem=$m $(date -Is)"
COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025" bash $T 2 8302 tSkywork7B-16k:Skywork/Skywork-OR1-Math-7B:16384 > $H/tl19_2b.out 2>&1 &
COMPS="custom/beyondaime" bash $T 0 8304 tSkywork7B-16kb:Skywork/Skywork-OR1-Math-7B:16384 > $H/tl19_0b.out 2>&1 &
COMPS="hmmt/hmmt_feb_2026 custom/amc" bash $T 3 8305 tSkywork7B-16kc:Skywork/Skywork-OR1-Math-7B:16384 tQ4BBaseGRPO-8k:Thinking-Space/Qwen3-4B-Base-GRPO:8192:32768 > $H/tl19_3b.out 2>&1 &
wait
echo "DRIVER19V_DONE $(date -Is)"
