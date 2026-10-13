#!/bin/bash
# GPU 0, after the 8B evals: offline pre-probe of Keven16 Math-Step500 as teacher for Qwen3-1.7B on the 1000 DAPO rollouts
until grep -q "EVAL_Q8B_DONE" /home/USER/logs/evq8b.log 2>/dev/null; do sleep 120; done
set -u; source /home/USER/opd/scripts/topk_common.sh
S=$OPD_HOME/scripts; G=0; RT=q3i17-dapo; TEACHER=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
dump $G $RT $TEACHER preq3i_TK4Bmath500
met preq3i_TQ3B4Bgrpo_student preq3i_TK4Bmath500 "pre Q3-1.7B vs Math500 (dapo)"
export CUDA_VISIBLE_DEVICES=$G
[ -s $F/ent_TK4Bmath500_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $TEACHER --rollouts $F/roll_$RT.jsonl --out $F/ent_TK4Bmath500_$RT.npz > $L/ent_TK4Bmath500_$RT.log 2>&1
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_TK4Bmath500_$RT.npz --label "pre Q3-1.7B vs Math500 (dapo)"
echo "PRE_Q3I_M500_DONE $(date -Is)"
