#!/bin/bash
# 19 GPU 3: top-64 ids/log-probs + full-vocab entropy of student / A / B on the 1000 DAPO rollouts (D0 diagnostics)
source /home/sya225/opd/env.sh none; export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3
R=/home/sya225/opd/fork/roll_q3i17-dapo.jsonl; O=/localscratch/sya225/fork_topk
for spec in "Qwen/Qwen3-1.7B tp_student" "Qwen/Qwen3-4B tp_T4B" "Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 tp_TM500"; do
  set -- $spec; echo "== $2 $(date -Is)"
  [ -s $O/$2_k64.npz ] || $VERL_VENV/bin/python /home/sya225/opd/scripts/fork_topk.py --model $1 --rollouts $R --out $O/$2_k64.npz --k 64 2>&1 | tail -1
done
echo "DUMP_TP_DONE $(date -Is)"
