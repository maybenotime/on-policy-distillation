#!/bin/bash
# cs-venus-05 GPUs 0,1,2 after the 4B resume: Code-Step1200 only (Math-500 moved
# to cs-venus-19, which steps ~3x faster; 05 expires 9/3 12:10).
set -u
sed -n '/^COMMON=/,/^}/p' /localscratch/sya225/chain_teachers.sh > /tmp/chainB_common.sh
source /tmp/chainB_common.sh
until grep -q "END 4b(resume)" /localscratch/sya225/logs/resume_4b_wrapper.log 2>/dev/null; do sleep 120; done
sleep 60
run K4Bcode1200 Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 0,1,2
echo "CHAIN_B2_DONE $(date -Is)"
