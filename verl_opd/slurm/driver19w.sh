#!/bin/bash
# Second split of the Skywork eval: lane c (hmmt26 + amc, ETA 12 h) -> hmmt26 resumed on GPU 3, amc as a new sibling on GPU 2;
# 4B-Base-GRPO (queued behind c) starts now on the idle GPU 1.  Lane b (beyondaime, GPU 0) is untouched.
H=/home/sya225/logs; T=/home/sya225/slurm/teacher_lane2.sh
echo "DRIVER19W_START $(date -Is)"
for pat in "teacher_lane2.sh 3 8305" "teacher_eval.sh tSkywork7B-16kc" "tSkywork7B_16kc" "--port 8305"; do pkill -f -- "$pat" 2>/dev/null; done
sleep 20; for pat in "tSkywork7B_16kc" "--port 8305"; do pkill -9 -f -- "$pat" 2>/dev/null; done
for i in $(seq 1 40); do m=$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 3000 ] && break; sleep 15; done
echo "GPU3_FREE mem=$m $(date -Is)"
COMPS="hmmt/hmmt_feb_2026" bash $T 3 8305 tSkywork7B-16kc:Skywork/Skywork-OR1-Math-7B:16384 > $H/tl19_3c.out 2>&1 &
COMPS="custom/amc"         bash $T 2 8306 tSkywork7B-16kd:Skywork/Skywork-OR1-Math-7B:16384 > $H/tl19_2c.out 2>&1 &
bash $T 1 8307 tQ4BBaseGRPO-8k:Thinking-Space/Qwen3-4B-Base-GRPO:8192:32768 > $H/tl19_1c.out 2>&1 &
wait
echo "DRIVER19W_DONE $(date -Is)"
