#!/bin/bash
# 19 GPU 3: offline pre-OPD probe for DeepSeek-R1-Distill-Qwen-1.5B (student) vs hbx/JustRL-DeepSeek-1.5B (teacher)
# on 1000 DAPO prompts: overlap@16/32 (vLLM top-k dumps) and rho / ForkOverlap / dH (HF full-vocab entropy).
set -u; source /home/USER/opd/scripts/topk_common.sh
S=$OPD_HOME/scripts; G=3; RT=ds15b-dapo
STUDENT=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; TEACHER=hbx/JustRL-DeepSeek-1.5B
ROLL_DATA=/home/USER/opd/verl_data/dapo_math17k/train.parquet bash $S/post_topk.sh $RT $STUDENT $TEACHER $G preds_TJustRL 0.85
echo "==== entropy (HF full vocab) ===="
export CUDA_VISIBLE_DEVICES=$G
[ -s $F/ent_student_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $STUDENT --rollouts $F/roll_$RT.jsonl --out $F/ent_student_$RT.npz > $L/ent_student_$RT.log 2>&1
[ -s $F/ent_TJustRL_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $TEACHER --rollouts $F/roll_$RT.jsonl --out $F/ent_TJustRL_$RT.npz > $L/ent_TJustRL_$RT.log 2>&1
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_TJustRL_$RT.npz --label "pre ds1.5B vs JustRL"
echo "PRE_DS_DONE $(date -Is)"
