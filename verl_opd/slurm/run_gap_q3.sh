#!/bin/bash
# gap-restricted hidden probe, Qwen3-1.7B (non-thinking) student on DAPO: teachers Qwen3-8B and Math-Step500
source /home/sya225/opd/env.sh none; P=$CODE_EVAL_ENV/bin/python; D=/localscratch/sya225/fork_topk; F=/home/sya225/opd/fork; S=/home/sya225/opd/scripts/hidden_probe.py; R=/home/sya225/opd/results
STU=Qwen/Qwen3-1.7B; ROLL=$F/roll_q3i17-dapo.jsonl; SD=$D/preq3i_TQ3B4Bgrpo_student.npz
echo "GAPQ3_START $(date -Is)"
run () { echo "=== $1"; CUDA_VISIBLE_DEVICES=$2 $P $S --student $3 --rollouts $ROLL --student_dump $4 --teachers $5 --layers -4 -3 -2 -1 --max_pos 1500 --n_seq 1000 --gap_thr 0.5 $6 --out $R/hiddengap_q3_$1.json 2>&1 | grep -v "Warning\|\[hid\] [0-9]"; }
( run student 0 $STU $SD "Q8B=$D/preq3i_T8B.npz M500=$D/preq3i_TK4Bmath500.npz" "" ) > /home/sya225/slurm/gapq3_g0.out 2>&1 &
( run self_Q8B  1 Qwen/Qwen3-8B $D/preq3i_T8B.npz "Q8B=$D/preq3i_T8B.npz" "--gap_ref $SD"
  run self_M500 1 Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 $D/preq3i_TK4Bmath500.npz "M500=$D/preq3i_TK4Bmath500.npz" "--gap_ref $SD" ) > /home/sya225/slurm/gapq3_g1.out 2>&1 &
wait
cat /home/sya225/slurm/gapq3_g0.out /home/sya225/slurm/gapq3_g1.out
echo "GAPQ3_DONE $(date -Is)"
