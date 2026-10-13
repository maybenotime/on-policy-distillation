#!/bin/bash
# After chain19 (RLCR): Code-Step300 then Math-Step500, both with a 16k response
# window. Their own generations on these prompts run to p95 13k / >8k for 28%,
# so the 8k window used for the short teachers would bind and drive the student
# into non-terminating outputs (what happened with Math-1200 at 8k).
set -u
source /home/sya225/opd/env.sh none
until grep -q "CHAIN_19_DONE" /home/sya225/logs/chain19.log 2>/dev/null; do sleep 120; done
sleep 60
sed -n '/^COMMON=/,/^}/p' $OPD_HOME/scripts/chain19.sh > /tmp/chain19_common.sh
source /tmp/chain19_common.sh
COMMON+=(MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576)   # last wins in env
run K4Bcode300_16k  Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300  0,1,2
run K4Bmath500_16k  Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500  0,1,2
echo "CHAIN_19B_DONE $(date -Is)"
