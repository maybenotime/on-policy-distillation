#!/bin/bash
# 19 GPU 0 (free until the DS-7B s200 eval starts): offline pre-probe of Qwen3-8B as teacher for Qwen3-1.7B on the
# existing 1000 DAPO rollouts of Qwen3-1.7B (roll_q3i17-dapo.jsonl): overlap@16/32 + rho/FO/dH.
set -u; source /home/sya225/opd/scripts/topk_common.sh
S=$OPD_HOME/scripts; G=0; RT=q3i17-dapo; TEACHER=Qwen/Qwen3-8B
dump $G $RT $TEACHER preq3i_T8B
met preq3i_TQ3B4Bgrpo_student preq3i_T8B "pre Q3-1.7B vs 8B (dapo)"
export CUDA_VISIBLE_DEVICES=$G
[ -s $F/ent_T8B_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $TEACHER --rollouts $F/roll_$RT.jsonl --out $F/ent_T8B_$RT.npz > $L/ent_T8B_$RT.log 2>&1
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_T8B_$RT.npz --label "pre Q3-1.7B vs 8B (dapo)"
echo "PRE_Q3I8B_DONE $(date -Is)"
