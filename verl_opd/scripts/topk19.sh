#!/bin/bash
# Top-k dumps on cs-venus-19: pre-OPD (base 1.7B rollouts vs 4 teachers) + post-OPD RLCR pair.
set -u; source /home/sya225/opd/scripts/topk_common.sh
RLCR=resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step
( dump 0 Qwen3-1.7B Qwen/Qwen3-1.7B pre_student;  dump 0 Qwen3-1.7B Qwen/Qwen3-14B pre_T14B ) &
( dump 2 Qwen3-1.7B Qwen/Qwen3-8B   pre_T8B;      dump 2 Qwen3-1.7B Qwen/Qwen3-4B  pre_T4B ) &
( dump 3 Qwen3-1.7B $RLCR pre_TRLCR;
  dump 3 opd-1.7b-fromRLCR4B /localscratch/sya225/checkpoints/hf_opd_1.7b_fromRLCR4B_step424 post_RLCR_student ) &
( until grep -q "fromRLCR4B\] DONE" /home/sya225/logs/fork_rlcr.log; do sleep 30; done
  dump 1 opd-1.7b-fromRLCR4B $RLCR post_RLCR_teacher ) &
wait
echo "==== metrics (19) ===="
for t in T8B T4B T14B TRLCR; do met pre_student pre_$t "pre 1.7B vs $t"; done
met post_RLCR_student post_RLCR_teacher "post 1.7B<-RLCR"
echo "TOPK19_DONE $(date -Is)"
