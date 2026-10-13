#!/bin/bash
# hidden-state probe at n=1000 with a layer sweep, plus the teacher-self controls (each teacher's own hidden states
# predicting its own entropy on the same student rollouts) — GPU 0 and 1 in parallel.
source /home/sya225/opd/env.sh none; P=$CODE_EVAL_ENV/bin/python; D=/localscratch/sya225/fork_topk; F=/home/sya225/opd/fork; S=/home/sya225/opd/scripts/hidden_probe.py; R=/home/sya225/opd/results
STU=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; ROLL=$F/roll_ds15b-dapo.jsonl
TEACH="JustRL=$D/preds_TJustRL_teacher.npz Sky7B=$D/preds_TSky7B.npz DS7B=$D/preds_TDS7B.npz"
echo "HIDDEN_FULL_START $(date -Is)"
run () { echo "=== $1"; CUDA_VISIBLE_DEVICES=$2 $P $S --student $3 --rollouts $ROLL --student_dump $4 --teachers $5 --layers $6 --max_pos 1500 --n_seq 1000 --out $R/hidden_$1.json 2>&1 | grep -v "Warning\|\[hid\] [0-9]"; }
( run student_last4 0 $STU $D/preds_TJustRL_student.npz "$TEACH" "-4 -3 -2 -1"
  run student_last1 0 $STU $D/preds_TJustRL_student.npz "$TEACH" "-1"
  run student_mid   0 $STU $D/preds_TJustRL_student.npz "$TEACH" "-14 -13 -12 -11" ) > /home/sya225/slurm/hidden_g0.out 2>&1 &
( run self_JustRL 1 hbx/JustRL-DeepSeek-1.5B $D/preds_TJustRL_teacher.npz "JustRL=$D/preds_TJustRL_teacher.npz" "-4 -3 -2 -1"
  run self_Sky7B  1 Skywork/Skywork-OR1-Math-7B $D/preds_TSky7B.npz "Sky7B=$D/preds_TSky7B.npz" "-4 -3 -2 -1"
  run self_DS7B   1 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B $D/preds_TDS7B.npz "DS7B=$D/preds_TDS7B.npz" "-4 -3 -2 -1" ) > /home/sya225/slurm/hidden_g1.out 2>&1 &
wait
cat /home/sya225/slurm/hidden_g0.out /home/sya225/slurm/hidden_g1.out
echo "HIDDEN_FULL_DONE $(date -Is)"
