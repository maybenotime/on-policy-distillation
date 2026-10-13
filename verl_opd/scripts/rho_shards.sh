#!/bin/bash
# Same as rho_worker.sh but processes an explicit list of shard indices, so a slow
# cell can be split across several GPUs.  Existing files are skipped, so two
# instances on disjoint shard lists never duplicate work.
#   rho_shards.sh <TAG> <FAMILY> <STUDENT> <DATA> <GPU> <TNAME> <TEACHER> <SHARD_CSV> [MAXTOK]
set -u
source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; S=$OPD_HOME/scripts; R=$SCRATCH/rho; L=$SCRATCH/logs
mkdir -p $R $L
export PATH="$E/bin:$PATH" TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING

TAG=$1; FAM=$2; STU=$3; DATA=$4; GPU=$5; TN=$6; TEA=$7; LIST=$8; MAXTOK=${9:-8192}
CH=${CHUNK:-256}
export CUDA_VISIBLE_DEVICES=$GPU

gpu_used () { nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1; }

roll () {  # $1=out  $2=log  $3=offset
  setsid "$E/bin/python" $S/rho_rollout.py --model "$STU" --family $FAM --data "$DATA" \
      --offset $3 --n 1000 --seed 0 --max_tokens $MAXTOK --out "$1" > "$2" 2>&1 &
  local pgid=$!
  for _ in $(seq 1 900); do
    grep -q '^\[rho\] wrote' "$2" 2>/dev/null && break
    kill -0 $pgid 2>/dev/null || break
    sleep 10
  done
  sleep 5; kill -9 -$pgid 2>/dev/null; kill -9 $pgid 2>/dev/null; wait $pgid 2>/dev/null
  for _ in $(seq 1 60); do [ "$(gpu_used)" -lt 3000 ] 2>/dev/null && break; sleep 5; done
}

echo "RHOS_START $TAG gpu=$GPU shards=$LIST $(date -Is)"
for k in ${LIST//,/ }; do
  RO=$R/roll_${TAG}_sh${k}.jsonl; LG=$L/rho_roll_${TAG}_sh${k}.log
  [ -s $RO ] || roll $RO $LG $((k*1000))
  [ -s $RO ] || { echo "  [$TAG sh$k] ROLLOUT FAILED"; continue; }
  for spec in "stu:$STU" "$TN:$TEA"; do
    NM=${spec%%:*}; MD=${spec#*:}
    OUT=$R/ent_${NM}_${TAG}_sh${k}.npz
    [ -s $OUT ] || "$E/bin/python" $S/fork_entropy.py --model "$MD" --rollouts $RO \
        --out $OUT --chunk $CH > $L/rho_ent_${NM}_${TAG}_sh${k}.log 2>&1
    [ -s $OUT ] || echo "  [$TAG sh$k] ENTROPY FAILED $NM"
  done
  echo "  [$TAG sh$k] done $(date -Is)"
  "$E/bin/python" $S/rho_agg.py --student "$R/ent_stu_${TAG}_sh*.npz" \
      --teacher "$R/ent_${TN}_${TAG}_sh*.npz" --label "$TAG vs $TN"
done
echo "RHOS_DONE $TAG $(date -Is)"
