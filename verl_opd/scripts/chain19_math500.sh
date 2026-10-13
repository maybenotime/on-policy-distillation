#!/bin/bash
# 19: wait for Code-300 16k resume to finish, then Math-500 16k on GPUs 0,1,2 (2 student + 1 teacher).
source /home/sya225/opd/env.sh none
until grep -q "RESUME_CODE300_DONE" /home/sya225/logs/resume_code300.log 2>/dev/null; do sleep 60; done
echo "code300 finished $(date -Is)"; sleep 60
sed -n '/^COMMON=/,/^}/p' $OPD_HOME/scripts/chain19.sh > /tmp/chain19_common.sh
source /tmp/chain19_common.sh
COMMON+=(MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576)
echo "######## START K4Bmath500_16k Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 gpus=0,1,2 $(date -Is) ########"
run K4Bmath500_16k Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 0,1,2
echo "######## END K4Bmath500_16k rc=$? $(date -Is) ########"
echo CHAIN_19_MATH_DONE
