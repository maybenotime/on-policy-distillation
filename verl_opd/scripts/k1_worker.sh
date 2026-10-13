#!/bin/bash
# k1_worker.sh <TAG> <STUDENT> <TEACHER> <GPU> <SHARD_CSV> [MAX_POS]
set -u
source /home/USER/opd/env.sh none
E=$CODE_EVAL_ENV; S=$OPD_HOME/scripts; R=$SCRATCH/rho; L=$SCRATCH/logs
TAG=$1; STU=$2; TEA=$3; GPU=$4; LIST=$5; MP=${6:-2048}
export PATH="$E/bin:$PATH" CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false
echo "K1_START $TAG gpu=$GPU shards=$LIST max_pos=$MP $(date -Is)"
for k in ${LIST//,/ }; do
  RO=$R/roll_${TAG}_sh${k}.jsonl; OUT=$R/k1_${TAG}_sh${k}.npz
  [ -s $RO ] || { echo "  missing $RO"; continue; }
  [ -s $OUT ] || "$E/bin/python" $S/k1_pos.py --student "$STU" --teacher "$TEA" \
      --rollouts $RO --out $OUT --k 16 --max_pos $MP > $L/k1_${TAG}_sh${k}.log 2>&1
  [ -s $OUT ] && echo "  [$TAG sh$k] ok $(date -Is)" || echo "  [$TAG sh$k] FAILED"
done
echo "K1_DONE $TAG $(date -Is)"
