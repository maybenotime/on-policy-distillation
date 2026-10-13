#!/bin/bash
# 19 GPU 3: post-OPD entropy dumps for the Code-300 (16k) student vs its teacher, on the student's own post rollouts
set -u; source /home/sya225/opd/scripts/topk_common.sh; S=$OPD_HOME/scripts; export CUDA_VISIBLE_DEVICES=3
T=opd-1.7b-fromK4Bcode300-16k; ST=$SCRATCH/checkpoints/hf_opd_1.7b_fromK4Bcode300_16k_step424; TE=Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300
[ -s $F/ent_student_$T.npz ] || "$E/bin/python" $S/fork_entropy.py --model $ST --rollouts $F/roll_$T.jsonl --out $F/ent_student_$T.npz > $L/ent_student_$T.log 2>&1
[ -s $F/ent_teacher_$T.npz ] || "$E/bin/python" $S/fork_entropy.py --model $TE --rollouts $F/roll_$T.jsonl --out $F/ent_teacher_$T.npz > $L/ent_teacher_$T.log 2>&1
"$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$T.npz --teacher_ent $F/ent_teacher_$T.npz --label "post $T"
echo "POST_ENT_C300_DONE $(date -Is)"
