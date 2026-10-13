#!/bin/bash
# Teacher benchmarks on cs-venus-05 job 270922 (3 lanes): the teachers cached only here.
H=/home/sya225/logs; T=/home/sya225/slurm/teacher_lane.sh
echo "DRIVER05Y_START $(date -Is)"
bash $T 0 8310 tNemotron-16k:nvidia/Nemotron-Research-Reasoning-Qwen-1.5B:16384 tQ8BGRPOMATH-8k:xiaoyuchen1/Qwen3-8B-GRPO-MATH:8192 > $H/tl05_0.out 2>&1 &
bash $T 1 8311 tDeepScaleR-16k:agentica-org/DeepScaleR-1.5B-Preview:16384 tQ8BDAPO-8k:pb09204048/Qwen3-8B-DAPO-iter449-disable-thinking:8192 > $H/tl05_1.out 2>&1 &
wait
echo "DRIVER05Y_DONE $(date -Is)"
