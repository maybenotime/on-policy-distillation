#!/bin/bash
# 19 GPU 0: post-OPD offline probe for DeepSeek-R1-Distill-Qwen-1.5B <- hbx/JustRL-DeepSeek-1.5B.
# Student = the trained s200 checkpoint, rolling out on its own; teacher scores the same 1000 DAPO
# rollouts. Same prompts/seed/max_tokens as the pre-probe (pre tag: ds15b-dapo), so pre and post compare.
set -u; source /home/sya225/opd/scripts/topk_common.sh
S=$OPD_HOME/scripts; G=0; RT=ds15b-fromJustRL-post
STUDENT=$SCRATCH/checkpoints/hf_ds15b_fromJustRL_step200; TEACHER=hbx/JustRL-DeepSeek-1.5B
[ -s "$STUDENT/config.json" ] || { echo "MISSING CHECKPOINT $STUDENT"; exit 1; }
ROLL_DATA=/home/sya225/opd/verl_data/dapo_math17k/train.parquet bash $S/post_topk.sh $RT $STUDENT $TEACHER $G preds_post_JustRL 0.85
echo "==== entropy (HF full vocab) ===="
export CUDA_VISIBLE_DEVICES=$G
[ -s $F/ent_student_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $STUDENT --rollouts $F/roll_$RT.jsonl --out $F/ent_student_$RT.npz > $L/ent_student_$RT.log 2>&1
[ -s $F/ent_teacher_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $TEACHER --rollouts $F/roll_$RT.jsonl --out $F/ent_teacher_$RT.npz > $L/ent_teacher_$RT.log 2>&1
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_teacher_$RT.npz --label "post ds1.5B <- JustRL"
echo "POST_DS_DONE JustRL $(date -Is)"
