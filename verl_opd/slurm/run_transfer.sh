#!/bin/bash
source /home/sya225/opd/env.sh none; P=$CODE_EVAL_ENV/bin/python; D=/localscratch/sya225/fork_topk; F=/home/sya225/opd/fork; S=/home/sya225/opd/scripts/transfer_probe.py
ls $F/roll_*.jsonl | tr '\n' ' '; echo
run () { [ -s "$2" ] && [ -s "$3" ] && [ -s "$4" ] || { echo "MISSING for $1: $2 $3 $4"; return; }; $P $S --student $2 --teacher $3 --rollouts $4 --label $1 2>&1 | tail -1; }
R=$F/roll_ds15b-dapo.jsonl
run DS.JustRL  $D/preds_TJustRL_student.npz $D/preds_TJustRL_teacher.npz $R
run DS.Sky7B   $D/preds_TJustRL_student.npz $D/preds_TSky7B.npz $R
run DS.DS7B    $D/preds_TJustRL_student.npz $D/preds_TDS7B.npz $R
R=$F/roll_q3i17-dapo.jsonl
run Q3.M500    $D/preq3i_TQ3B4Bgrpo_student.npz $D/preq3i_TK4Bmath500.npz $R
run Q3.8B      $D/preq3i_TQ3B4Bgrpo_student.npz $D/preq3i_T8B.npz $R
run Q3.BaseGRPO $D/preq3i_TQ3B4Bgrpo_student.npz $D/preq3i_TQ3B4Bgrpo_teacher.npz $R
R=$F/roll_Qwen3-1.7B.jsonl
for T in C300:pre_TK4Bcode300 C1200:pre_TK4Bcode1200 RLCR:pre_TRLCR 8B:pre_T8B 4B:pre_T4B 14B:pre_T14B; do run code.${T%%:*} $D/pre_student.npz $D/${T##*:}.npz $R; done
echo "TRANSFER_DONE $(date -Is)"
