#!/bin/bash
# 05 (job 264439, 3 GPUs): Math-1200 16k from scratch, 2 student + 1 teacher on GPUs 0,1,2.
source /home/USER/opd/env.sh none
sed -n '/^COMMON=(/,/^}/p' /home/USER/opd/scripts/chain05.sh > /tmp/chain05_common.sh
source /tmp/chain05_common.sh
COMMON+=(MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576)
rm -rf $SCRATCH/checkpoints/opd_code_17b_fromK4Bmath1200_16k /tmp/ray_K4Bmath1200_16k
mv $SCRATCH/logs/train_17b_fromK4Bmath1200_16k.log $SCRATCH/logs/train_17b_fromK4Bmath1200_16k.timeout98.log 2>/dev/null
run K4Bmath1200_16k Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200 0,1,2 2 1
echo CHAIN_05_MATH_DONE
