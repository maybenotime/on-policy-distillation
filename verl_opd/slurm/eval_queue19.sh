#!/bin/bash
# GPU-3 eval queue on cs-venus-19: waits for the pair-2 generation to finish, then evaluates s200 and s50 of every queued run.
J=269301; SL=/home/sya225/slurm; Q=$SL/eval_queue19.txt; PORT=8120; n=0
until grep -q CHAIN_ALL_DONE $SL/deepmath_chain4.out 2>/dev/null; do sleep 600; done
echo "EVQ_START $(date -Is)"
while true; do
  line=$(sed -n "$((n+1))p" $Q); if [ -z "$line" ]; then grep -q DRIVER19B_ALL_TRAINED $SL/driver19b.out 2>/dev/null && [ "$n" -ge "$(wc -l < $Q)" ] && break; sleep 300; continue; fi
  set -- $line; TAG=$1; NAME=$2; BASE=$3; MT=$4; n=$((n+1)); PORT=$((PORT+2))
  srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c "bash $SL/eval19.sh $TAG 100 3 $PORT $NAME $BASE $MT; bash $SL/eval19.sh $TAG 50 3 $((PORT+1)) $NAME $BASE $MT" 2>&1 | grep -v "module avail\|module load\|teamdynamix"
done
echo "EVQ_ALL_DONE $(date -Is)"
