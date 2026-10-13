#!/bin/bash
# login-node watcher: stop pair 2 (Q3_8B + M500) once both_wrong >= 7000 (both_correct already >= 10000).
SL=/home/sya225/slurm; OUT=/home/sya225/opd/verl_data/deepmath; KW=7000; KC=10000
while true; do
  P=$SL/dm_peek_Q3_8B_M500.txt
  bc=$(grep -o "both correct: [0-9]*" $P 2>/dev/null | grep -o "[0-9]*$"); bw=$(grep -o "both wrong: [0-9]*" $P 2>/dev/null | grep -o "[0-9]*$")
  echo "$(date -Is) both_correct=${bc:-?} both_wrong=${bw:-?}"
  if [ "${bc:-0}" -ge $KC ] && [ "${bw:-0}" -ge $KW ]; then
    touch $OUT/STOP_Q3_8B $OUT/STOP_M500; echo "STOP7K_TOUCHED $(date -Is)"; break
  fi
  grep -q "PAIR Q3_8B + M500 GEN DONE" $SL/deepmath_chain2.out 2>/dev/null && { echo "pair already finished"; break; }
  sleep 600
done
