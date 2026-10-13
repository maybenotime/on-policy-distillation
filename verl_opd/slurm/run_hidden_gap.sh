#!/bin/bash
# gap-restricted hidden probe: student side (GPU 0) and teacher-self ceilings on the same gap positions (GPU 1)
source /home/sya225/opd/env.sh none; P=$CODE_EVAL_ENV/bin/python; D=/localscratch/sya225/fork_topk; F=/home/sya225/opd/fork; S=/home/sya225/opd/scripts/hidden_probe.py; R=/home/sya225/opd/results
STU=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; ROLL=$F/roll_ds15b-dapo.jsonl; SD=$D/preds_TJustRL_student.npz
echo "GAP_START $(date -Is)"
run () { echo "=== $1"; CUDA_VISIBLE_DEVICES=$2 $P $S --student $3 --rollouts $ROLL --student_dump $4 --teachers $5 --layers -4 -3 -2 -1 --max_pos 1500 --n_seq 1000 --gap_thr 0.5 $6 --out $R/hiddengap_$1.json 2>&1 | grep -v "Warning\|\[hid\] [0-9]"; }
( run student 0 $STU $SD "JustRL=$D/preds_TJustRL_teacher.npz Sky7B=$D/preds_TSky7B.npz DS7B=$D/preds_TDS7B.npz" "" ) > /home/sya225/slurm/hiddengap_g0.out 2>&1 &
( run self_JustRL 1 hbx/JustRL-DeepSeek-1.5B $D/preds_TJustRL_teacher.npz "JustRL=$D/preds_TJustRL_teacher.npz" "--gap_ref $SD"
  run self_Sky7B  1 Skywork/Skywork-OR1-Math-7B $D/preds_TSky7B.npz "Sky7B=$D/preds_TSky7B.npz" "--gap_ref $SD"
  run self_DS7B   1 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B $D/preds_TDS7B.npz "DS7B=$D/preds_TDS7B.npz" "--gap_ref $SD" ) > /home/sya225/slurm/hiddengap_g1.out 2>&1 &
wait
cat /home/sya225/slurm/hiddengap_g0.out /home/sya225/slurm/hiddengap_g1.out
echo "GAP_DONE $(date -Is)"
