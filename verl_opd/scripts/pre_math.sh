#!/bin/bash
# job 264439 GPU 2 (shared with the running teacher vLLM, ~50 GB free): pre-OPD metrics on math prompts.
# 1) base-1.7B rollouts on ot3_math30k (via post_topk.sh, util 0.3) + student/teacher top-k for Math-1200,
# 2) top-k for the other teachers, 3) full-vocab entropy (rho / ForkOverlap / dH) for student + teachers.
set -u; source /home/sya225/opd/scripts/topk_common.sh
S=$OPD_HOME/scripts; G=2; RT=Qwen3-1.7B-math
ROLL_DATA=/home/sya225/opd/verl_data/ot3_math30k/train.parquet bash $S/post_topk.sh $RT Qwen/Qwen3-1.7B Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200 $G premath_TK4Bmath1200 0.3
ln -sf $D/premath_TK4Bmath1200_student.npz $D/premath_student.npz
dump $G $RT Qwen/Qwen3-4B                                     premath_T4B
dump $G $RT Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200    premath_TK4Bcode1200
dump $G $RT Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500     premath_TK4Bmath500
dump $G $RT Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300     premath_TK4Bcode300
dump $G $RT resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step                             premath_TRLCR4B
echo "==== overlap metrics (math prompts) ===="
for t in TK4Bmath1200 T4B TK4Bcode1200 TK4Bmath500 TK4Bcode300 TRLCR4B; do met premath_student premath_$t "pre-math 1.7B vs $t"; done
echo "==== entropy ===="
export CUDA_VISIBLE_DEVICES=$G
[ -s $F/ent_student_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model Qwen/Qwen3-1.7B --rollouts $F/roll_$RT.jsonl --out $F/ent_student_$RT.npz > $L/ent_student_$RT.log 2>&1
for pair in TK4Bmath1200:Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200 T4B:Qwen/Qwen3-4B TK4Bcode1200:Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 TK4Bmath500:Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 TK4Bcode300:Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300 TRLCR4B:resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step; do
  t=${pair%%:*}; m=${pair#*:}
  [ -s $F/ent_${t}_$RT.npz ] || "$E/bin/python" $S/fork_entropy.py --model $m --rollouts $F/roll_$RT.jsonl --out $F/ent_${t}_$RT.npz > $L/ent_${t}_$RT.log 2>&1
  "$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_${t}_$RT.npz --label "pre-math 1.7B vs $t"
done
echo "PRE_MATH_DONE $(date -Is)"
