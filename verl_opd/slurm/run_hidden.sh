#!/bin/bash
# waits for the reachability probe to release GPU 1, then runs the hidden-state (representational sufficiency) probe
source /home/sya225/opd/env.sh none; P=$CODE_EVAL_ENV/bin/python; D=/localscratch/sya225/fork_topk; F=/home/sya225/opd/fork
until grep -q REACH_DONE /home/sya225/slurm/reach_probe.out 2>/dev/null; do sleep 120; done
echo "HIDDEN_START $(date -Is)"
CUDA_VISIBLE_DEVICES=1 $P /home/sya225/opd/scripts/hidden_probe.py --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --rollouts $F/roll_ds15b-dapo.jsonl --student_dump $D/preds_TJustRL_student.npz \
  --teachers JustRL=$D/preds_TJustRL_teacher.npz Sky7B=$D/preds_TSky7B.npz DS7B=$D/preds_TDS7B.npz \
  --layers -4 -3 -2 -1 --max_pos 1500 --out /home/sya225/opd/results/hidden_probe_ds.json 2>&1 | grep -v Warning
echo "HIDDEN_END $(date -Is)"
