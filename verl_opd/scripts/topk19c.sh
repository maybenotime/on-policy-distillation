#!/bin/bash
# 19 GPU 3: pre-OPD top-k dumps for the two RL-code teachers on the base-1.7B rollouts.
set -u; source /home/USER/opd/scripts/topk_common.sh
dump 3 Qwen3-1.7B Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300  pre_TK4Bcode300
dump 3 Qwen3-1.7B Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 pre_TK4Bcode1200
echo "==== metrics (19c) ===="
for t in TK4Bcode300 TK4Bcode1200 T4B; do met pre_student pre_$t "pre 1.7B vs $t"; done
echo "TOPK19C_DONE $(date -Is)"
