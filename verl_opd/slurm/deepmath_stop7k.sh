#!/bin/bash
# login-node watcher: stop pair 1 (JustRL + DS7B) once both_wrong >= 7000 (both_correct already >= 10000).
SL=/home/sya225/slurm; OUT=/home/sya225/opd/verl_data/deepmath; KW=7000; KC=10000
while true; do
  P=$SL/dm_peek_JustRL_DS7B.txt
  bc=$(grep -o "both correct: [0-9]*" $P 2>/dev/null | grep -o "[0-9]*$"); bw=$(grep -o "both wrong: [0-9]*" $P 2>/dev/null | grep -o "[0-9]*$")
  echo "$(date -Is) both_correct=${bc:-?} both_wrong=${bw:-?}"
  if [ "${bc:-0}" -ge $KC ] && [ "${bw:-0}" -ge $KW ]; then
    touch $OUT/STOP_JustRL $OUT/STOP_DS7B; echo "STOP7K_TOUCHED $(date -Is)"; break
  fi
  grep -q "PAIR JustRL + DS7B GEN DONE" $SL/deepmath_chain.out 2>/dev/null && { echo "pair already finished"; break; }
  sleep 600
done
