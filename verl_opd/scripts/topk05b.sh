#!/bin/bash
# 4B<-8B pair, post side (opd-4b = hf_opd30k_step424 lives on 05), after topk05.sh.
set -u; source /home/sya225/opd/scripts/topk_common.sh
until grep -q TOPK05_DONE /home/sya225/logs/topk05.log; do sleep 30; done
dump 0 opd-4b /localscratch/sya225/checkpoints/hf_opd30k_step424 post4B_student
dump 0 opd-4b Qwen/Qwen3-8B post4B_T8B
met post4B_student post4B_T8B "post 4B<-T8B"
echo "TOPK05B_DONE $(date -Is)"
