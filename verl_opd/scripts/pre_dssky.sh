#!/bin/bash
# 19 GPU 0: offline pre-probe of Skywork-OR1-Math-7B as teacher for R1-Distill-1.5B on the existing 1000 DAPO rollouts
set -u; source /home/USER/opd/scripts/topk_common.sh
S=$OPD_HOME/scripts; G=0; RT=ds15b-dapo; TEACHER=Skywork/Skywork-OR1-Math-7B
dump $G $RT $TEACHER preds_TSky7B
met preds_TJustRL_student preds_TSky7B "pre ds1.5B vs Skywork-OR1-7B"
export CUDA_VISIBLE_DEVICES=$G
[ -s $F/ent_TSky7B_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $TEACHER --rollouts $F/roll_$RT.jsonl --out $F/ent_TSky7B_$RT.npz > $L/ent_TSky7B_$RT.log 2>&1
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_TSky7B_$RT.npz --label "pre ds1.5B vs Skywork-OR1-7B"
echo "PRE_DSSKY_DONE $(date -Is)"
