#!/bin/bash
set -x
W=/home/sya225/opd/scripts/eval_v5.sh; L=/localscratch/sya225/logs
bash $W qwen3_8b   Qwen/Qwen3-8B                                     Qwen3-8B   8010 0,5 2 > $L/v5_Qwen3-8B.log   2>&1 &
bash $W qwen3_4b   Qwen/Qwen3-4B                                     Qwen3-4B   8011 1   1 > $L/v5_Qwen3-4B.log   2>&1 &
bash $W qwen3_1p7b Qwen/Qwen3-1.7B                                   Qwen3-1.7B 8012 2   1 > $L/v5_Qwen3-1.7B.log 2>&1 &
bash $W opd_4b     /localscratch/sya225/checkpoints/hf_opd30k_step424 opd-4b     8013 3   1 > $L/v5_opd-4b.log    2>&1 &
bash $W opd_1p7b   /localscratch/sya225/checkpoints/hf_opd17b_step424 opd-1.7b   8014 4   1 > $L/v5_opd-1.7b.log  2>&1 &
wait
echo "ALL_DONE $(date -Is)"
