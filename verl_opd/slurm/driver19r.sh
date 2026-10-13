#!/bin/bash
# Training queue on cs-venus-19 GPUs 0,1,2 (after q3i17_ts_mismatchW finishes training):
#   1. q3i17_tpM500_shapeStuFix : M500 preferences + FROZEN base-student (Qwen3-1.7B) entropy profile, 100 steps
#   2. ds15b_tpJustRL_shapeStuFix: JustRL preferences + FROZEN base-student (DS-R1-1.5B) profile, 200 steps
# Both are TRANSPLANT_MODE=shape with B = the base student, replacing the 'student' mode runs whose target moved
# with the training student.  driver19q's own evals are killed and handed to driver19s.sh (GPU 3) so the GPUs stay
# on training.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
M5=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500; Q3=Qwen/Qwen3-1.7B
JR=hbx/JustRL-DeepSeek-1.5B; DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
echo "DRIVER19R_START $(date -Is)"
until grep -q TRAIN_RC_TSW $H/driver19q.out 2>/dev/null; do sleep 30; done
echo "TSW_TRAIN_ENDED $(date -Is)"
pkill -f "bash /home/sya225/slurm/driver19q.sh"; sleep 2
pkill -f "hf_q3i17_ts_mismatchW"; pkill -f "teacher_eval.sh q3TS_mismatchW"; sleep 5
pkill -9 -f "hf_q3i17_ts_mismatchW" 2>/dev/null
rm -rf $SCRATCH/checkpoints/hf_q3i17_ts_mismatchW_step200   # possibly half-merged by the killed driver; driver19s re-merges
for i in $(seq 1 80); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2 | sort -n | tail -1); [ "$m" -lt 3000 ] && break; sleep 15
done
echo "GPUS_FREE mem=$m $(date -Is)"
SAVE_FREQ=50 bash $S/chain19_tp.sh q3i17_tpM500_shapeStuFix shape $M5 $Q3 0,1,2 /tmp/ray_m500fix 1.0 100
echo "TRAIN_RC_M500FIX=$? $(date -Is)"
sleep 60
: DS rerun moved to node 05 - driver05x.sh - at user request 9-21 23:25                                             
echo "TRAIN_RC_DSFIX=$? $(date -Is)"
echo "DRIVER19R_DONE $(date -Is)"
