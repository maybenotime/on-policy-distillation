#!/bin/bash
# Top-k dumps on node-A (GPU 0 only, after the from4b fork worker frees it):
# post-OPD pairs whose merged checkpoints live on this node.
set -u; source /home/USER/opd/scripts/topk_common.sh
until grep -q "from4b\] DONE" /home/USER/logs/fork_from4b.log; do sleep 30; done
C=/localscratch/USER/checkpoints
dump 0 opd-1.7b-from4b   $C/hf_opd_1.7b_from4b_step424   post_4B_student
dump 0 opd-1.7b-from4b   Qwen/Qwen3-4B                   post_4B_teacher
dump 0 opd-1.7b          $C/hf_opd17b_step424            post_8B_student
dump 0 opd-1.7b          Qwen/Qwen3-8B                   post_8B_teacher
dump 0 opd-1.7b-from14b  $C/hf_opd17b_from14b_step424    post_14B_student
dump 0 opd-1.7b-from14b  Qwen/Qwen3-14B                  post_14B_teacher
echo "==== metrics (05) ===="
for t in 4B 8B 14B; do met post_${t}_student post_${t}_teacher "post 1.7B<-$t"; done
echo "TOPK05_DONE $(date -Is)"
