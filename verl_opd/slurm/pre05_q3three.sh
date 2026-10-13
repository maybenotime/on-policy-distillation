#!/bin/bash
# cs-venus-05: pre-probes (4k/T0.7, same protocol as the existing Qwen3 rows) of three teachers vs Qwen3-1.7B non-thinking
# on the existing 1000 DAPO rollouts (roll_q3i17-dapo.jsonl).  bash pre05_q3three.sh <GPU>
set -u; source /home/sya225/opd/scripts/topk_common.sh; S=$OPD_HOME/scripts; RT=q3i17-dapo; G=${1:-2}
STU=Qwen/Qwen3-1.7B
echo "PREQ3_START $(date -Is)"
dump $G $RT $STU preds_student_q3i17
probe () {  # name model
  dump $G $RT "$2" preds_T$1
  met preds_student_q3i17 preds_T$1 "pre q3-1.7B vs $1"
  [ -s $F/ent_T$1_$RT.npz ] || CUDA_VISIBLE_DEVICES=$G "$E/bin/python" $S/fork_entropy.py --model "$2" --rollouts $F/roll_$RT.jsonl --out $F/ent_T$1_$RT.npz > $L/ent_T$1_$RT.log 2>&1
  "$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_T$1_$RT.npz --label "pre q3-1.7B vs $1"
  "$E/bin/python" $S/transfer_probe.py --student $D/preds_student_q3i17.npz --teacher $D/preds_T$1.npz --rollouts $F/roll_$RT.jsonl --label Q3.$1 2>&1 | tail -1
  echo "PROBE_DONE $1 $(date -Is)"
}
probe Q8Bdapo pb09204048/Qwen3-8B-DAPO-iter449-disable-thinking
probe K4Bcode1200 Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200
probe Q8Bgrpomath xiaoyuchen1/Qwen3-8B-GRPO-MATH
echo "PREQ3_DONE $(date -Is)"
