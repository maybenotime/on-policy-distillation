#!/bin/bash
# Teacher benchmarks on cs-venus-19 (4 lanes).  DS-lineage teachers at 16k (thinking), Qwen3 teachers at 8k (non-thinking).
H=/home/sya225/logs; T=/home/sya225/slurm/teacher_lane.sh
echo "DRIVER19U_START $(date -Is)"
bash $T 0 8300 tJustRL-16k:hbx/JustRL-DeepSeek-1.5B:16384 tM500-8k:Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500:8192 > $H/tl19_0.out 2>&1 &
bash $T 1 8301 tR1D7B-16k:deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:16384 tK4Bcode1200-8k:Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200:8192 > $H/tl19_1.out 2>&1 &
bash $T 2 8302 tSkywork7B-16k:Skywork/Skywork-OR1-Math-7B:16384 tQ4BBaseGRPO-8k:Thinking-Space/Qwen3-4B-Base-GRPO:8192:32768 > $H/tl19_2.out 2>&1 &
bash $T 3 8303 tLightR1-16k:qihoo360/Light-R1-7B-DS:16384 > $H/tl19_3.out 2>&1 &
wait
echo "DRIVER19U_DONE $(date -Is)"
