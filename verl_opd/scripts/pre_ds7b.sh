#!/bin/bash
# 19 GPU 3: offline pre-probe of DeepSeek-R1-Distill-Qwen-7B as teacher for R1-Distill-1.5B, on the existing 1000 DAPO rollouts
set -u; source /home/sya225/opd/scripts/topk_common.sh
S=$OPD_HOME/scripts; G=3; RT=ds15b-dapo; TEACHER=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
dump $G $RT $TEACHER preds_TDS7B
met preds_TJustRL_student preds_TDS7B "pre ds1.5B vs DS7B"
met preds_TJustRL_student preds_TJustRL_teacher "pre ds1.5B vs JustRL"
export CUDA_VISIBLE_DEVICES=$G
[ -s $F/ent_TDS7B_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $TEACHER --rollouts $F/roll_$RT.jsonl --out $F/ent_TDS7B_$RT.npz > $L/ent_TDS7B_$RT.log 2>&1
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_TDS7B_$RT.npz --label "pre ds1.5B vs DS7B"
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_TJustRL_$RT.npz --label "pre ds1.5B vs JustRL"
echo "PRE_DS7B_DONE $(date -Is)"
