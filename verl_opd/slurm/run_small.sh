#!/bin/bash
bash /home/sya225/slurm/reach_probe2.sh 50
source /home/sya225/opd/env.sh none; P=$CODE_EVAL_ENV/bin/python; D=/localscratch/sya225/fork_topk; F=/home/sya225/opd/fork
echo "HIDDEN_START n=200 $(date -Is)"
CUDA_VISIBLE_DEVICES=1 $P /home/sya225/opd/scripts/hidden_probe.py --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --rollouts $F/roll_ds15b-dapo.jsonl --student_dump $D/preds_TJustRL_student.npz \
  --teachers JustRL=$D/preds_TJustRL_teacher.npz Sky7B=$D/preds_TSky7B.npz DS7B=$D/preds_TDS7B.npz \
  --layers -4 -3 -2 -1 --max_pos 1500 --n_seq 200 --out /home/sya225/opd/results/hidden_probe_ds_n200.json 2>&1 | grep -v Warning
echo "HIDDEN_END $(date -Is)"
