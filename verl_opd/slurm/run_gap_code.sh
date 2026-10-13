#!/bin/bash
# gap-restricted hidden probe, Qwen3-1.7B student on CODE prompts: six teachers (waits for the Qwen3-DAPO group)
source /home/sya225/opd/env.sh none; P=$CODE_EVAL_ENV/bin/python; D=/localscratch/sya225/fork_topk; F=/home/sya225/opd/fork; S=/home/sya225/opd/scripts/hidden_probe.py; R=/home/sya225/opd/results
until grep -q GAPQ3_DONE /home/sya225/slurm/run_gap_q3.out 2>/dev/null; do sleep 120; done
STU=Qwen/Qwen3-1.7B; ROLL=$F/roll_Qwen3-1.7B.jsonl; SD=$D/pre_student.npz
echo "GAPCODE_START $(date -Is)"
run () { echo "=== $1"; CUDA_VISIBLE_DEVICES=$2 $P $S --student $3 --rollouts $ROLL --student_dump $4 --teachers $5 --layers -4 -3 -2 -1 --max_pos 1500 --n_seq 1000 --gap_thr 0.5 $6 --out $R/hiddengap_code_$1.json 2>&1 | grep -v "Warning\|\[hid\] [0-9]"; }
( run student 0 $STU $SD "C300=$D/pre_TK4Bcode300.npz C1200=$D/pre_TK4Bcode1200.npz RLCR=$D/pre_TRLCR.npz Q8B=$D/pre_T8B.npz Q4B=$D/pre_T4B.npz Q14B=$D/pre_T14B.npz" ""
  run self_C300  0 Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300  $D/pre_TK4Bcode300.npz  "C300=$D/pre_TK4Bcode300.npz"  "--gap_ref $SD"
  run self_C1200 0 Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 $D/pre_TK4Bcode1200.npz "C1200=$D/pre_TK4Bcode1200.npz" "--gap_ref $SD"
  run self_RLCR  0 resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step $D/pre_TRLCR.npz "RLCR=$D/pre_TRLCR.npz" "--gap_ref $SD" ) > /home/sya225/slurm/gapcode_g0.out 2>&1 &
( run self_Q8B  1 Qwen/Qwen3-8B  $D/pre_T8B.npz  "Q8B=$D/pre_T8B.npz"   "--gap_ref $SD"
  run self_Q4B  1 Qwen/Qwen3-4B  $D/pre_T4B.npz  "Q4B=$D/pre_T4B.npz"   "--gap_ref $SD"
  run self_Q14B 1 Qwen/Qwen3-14B $D/pre_T14B.npz "Q14B=$D/pre_T14B.npz" "--gap_ref $SD" ) > /home/sya225/slurm/gapcode_g1.out 2>&1 &
wait
cat /home/sya225/slurm/gapcode_g0.out /home/sya225/slurm/gapcode_g1.out
echo "GAPCODE_DONE $(date -Is)"
