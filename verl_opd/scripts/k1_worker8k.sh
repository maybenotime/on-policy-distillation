#!/bin/bash
# k1_worker8k.sh <TAG> <STUDENT> <TEACHER> <GPU> <SHARD_CSV>   -- full 8192-position sweep
set -u
source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; S=$OPD_HOME/scripts; R=$SCRATCH/rho; L=$SCRATCH/logs
TAG=$1; STU=$2; TEA=$3; GPU=$4; LIST=$5
export PATH="$E/bin:$PATH" CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false
echo "K18K_START $TAG gpu=$GPU shards=$LIST $(date -Is)"
for k in ${LIST//,/ }; do
  RO=$R/roll_${TAG}_sh${k}.jsonl; OUT=$R/k18k_${TAG}_sh${k}.npz
  [ -s $RO ] || { echo "  missing $RO"; continue; }
  [ -s $OUT ] || "$E/bin/python" $S/k1_pos.py --student "$STU" --teacher "$TEA" \
      --rollouts $RO --out $OUT --k 16 --max_pos 8192 > $L/k18k_${TAG}_sh${k}.log 2>&1
  [ -s $OUT ] && echo "  [$TAG sh$k] ok $(date -Is)" || echo "  [$TAG sh$k] FAILED"
done
echo "K18K_DONE $TAG $(date -Is)"
