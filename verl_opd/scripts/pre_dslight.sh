#!/bin/bash
# 19 GPU 0: offline pre-probe of qihoo360/Light-R1-7B-DS as teacher for R1-Distill-1.5B on the existing 1000 DAPO rollouts.
# Waits for the model download and for GPU 0 (Qwen3-8B DeepMath shard 0) to free up.
set -u; source /home/USER/opd/scripts/topk_common.sh
S=$OPD_HOME/scripts; G=0; RT=ds15b-dapo; TEACHER=qihoo360/Light-R1-7B-DS
until grep -q DL_DONE /home/USER/slurm/dl_light.out 2>/dev/null; do sleep 60; done
until grep -q GEN_END /home/USER/slurm/dm_Q3_8B_0.out 2>/dev/null; do sleep 120; done
sleep 30; echo "PRE_LIGHT_START $(date -Is)"
dump $G $RT $TEACHER preds_TLight7B
met preds_TJustRL_student preds_TLight7B "pre ds1.5B vs Light-R1-7B-DS"
export CUDA_VISIBLE_DEVICES=$G
[ -s $F/ent_TLight7B_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $TEACHER --rollouts $F/roll_$RT.jsonl --out $F/ent_TLight7B_$RT.npz > $L/ent_TLight7B_$RT.log 2>&1
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_TLight7B_$RT.npz --label "pre ds1.5B vs Light-R1-7B-DS"
"$E/bin/python" $S/transfer_probe.py --student $D/preds_TJustRL_student.npz --teacher $D/preds_TLight7B.npz --rollouts $F/roll_$RT.jsonl --label DS.Light7B 2>&1 | tail -1
echo "PRE_LIGHT_DONE $(date -Is)"
