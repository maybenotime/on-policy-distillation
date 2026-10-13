#!/bin/bash
# 19 GPU 3: offline pre-OPD probe, Qwen3-1.7B-Base (student) vs Thinking-Space/Qwen3-4B-Base-GRPO (teacher), 1000 DAPO prompts
set -u; source /home/sya225/opd/scripts/topk_common.sh
S=$OPD_HOME/scripts; G=3; RT=q3i17-dapo
STUDENT=Qwen/Qwen3-1.7B; TEACHER=Thinking-Space/Qwen3-4B-Base-GRPO
ROLL_DATA=/home/sya225/opd/verl_data/dapo_math17k/train.parquet bash $S/post_topk.sh $RT $STUDENT $TEACHER $G preq3i_TQ3B4Bgrpo 0.85
echo "==== entropy (HF full vocab) ===="
export CUDA_VISIBLE_DEVICES=$G
[ -s $F/ent_student_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $STUDENT --rollouts $F/roll_$RT.jsonl --out $F/ent_student_$RT.npz > $L/ent_student_$RT.log 2>&1
[ -s $F/ent_TQ3B4Bgrpo_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $TEACHER --rollouts $F/roll_$RT.jsonl --out $F/ent_TQ3B4Bgrpo_$RT.npz > $L/ent_TQ3B4Bgrpo_$RT.log 2>&1
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_TQ3B4Bgrpo_$RT.npz --label "pre Q3-1.7B vs 4B-Base-GRPO"
echo "PRE_Q3B_DONE $(date -Is)"
