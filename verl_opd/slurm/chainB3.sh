#!/bin/bash
# cs-venus-05: Code-Step1200 on the idle GPUs 3,4,5 right away (4B still finishing on 0-2).
set -u
sed -n '/^COMMON=/,/^}/p' /localscratch/sya225/chain_teachers.sh > /tmp/chainB_common.sh
source /tmp/chainB_common.sh
run K4Bcode1200 Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 3,4,5
echo "CHAIN_B3_DONE $(date -Is)"
