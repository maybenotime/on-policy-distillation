#!/bin/bash
# 4B<-8B pair, pre side (base Qwen3-4B rollouts) on 19.
set -u; source /home/USER/opd/scripts/topk_common.sh
( dump 0 Qwen3-4B Qwen/Qwen3-4B pre4B_student ) &
( dump 2 Qwen3-4B Qwen/Qwen3-8B pre4B_T8B ) &
wait
met pre4B_student pre4B_T8B "pre 4B vs T8B"
echo "TOPK19B_DONE $(date -Is)"
