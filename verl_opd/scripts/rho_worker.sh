#!/bin/bash
# One (student x dataset) cell: sharded rollout + full-vocab entropy for the student
# and one teacher.  1000 prompts per shard, 7 shards = the whole 7000-row subset.
#   rho_worker.sh <TAG> <FAMILY> <STUDENT> <DATA> <GPU> <TNAME> <TEACHER> [SHARDS] [MAXTOK]
set -u
source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; S=$OPD_HOME/scripts; R=$SCRATCH/rho; L=$SCRATCH/logs
mkdir -p $R $L
export PATH="$E/bin:$PATH" TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING

TAG=$1; FAM=$2; STU=$3; DATA=$4; GPU=$5; TN=$6; TEA=$7
SHARDS=${8:-7}; MAXTOK=${9:-8192}; CH=${CHUNK:-256}
export CUDA_VISIBLE_DEVICES=$GPU

gpu_free () {  # MiB currently used on our physical GPU
  nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

# vLLM 0.27 can hang in EngineCore teardown *after* the jsonl is complete.  The
# engine lives in a CHILD process, so the rollout runs under setsid and the whole
# process group is killed; then we wait for the GPU memory to actually come back
# before the HF forward passes start.
roll () {  # $1=out jsonl  $2=log  $3=offset
  setsid "$E/bin/python" $S/rho_rollout.py --model "$STU" --family $FAM --data "$DATA" \
      --offset $3 --n 1000 --seed 0 --max_tokens $MAXTOK --out "$1" > "$2" 2>&1 &
  local pgid=$!
  for _ in $(seq 1 900); do          # up to 2.5 h
    grep -q '^\[rho\] wrote' "$2" 2>/dev/null && break
    kill -0 $pgid 2>/dev/null || break
    sleep 10
  done
  sleep 5; kill -9 -$pgid 2>/dev/null; kill -9 $pgid 2>/dev/null; wait $pgid 2>/dev/null
  for _ in $(seq 1 60); do
    [ "$(gpu_free)" -lt 3000 ] 2>/dev/null && break
    sleep 5
  done
  echo "  [$TAG] gpu$GPU free after rollout: $(gpu_free) MiB"
}

echo "RHO_START $TAG gpu=$GPU teacher=$TN shards=$SHARDS maxtok=$MAXTOK $(date -Is)"
for k in $(seq 0 $((SHARDS-1))); do
  OFF=$((k*1000))
  RO=$R/roll_${TAG}_sh${k}.jsonl
  LG=$L/rho_roll_${TAG}_sh${k}.log
  [ -s $RO ] || roll $RO $LG $OFF
  if [ ! -s $RO ]; then echo "  [$TAG sh$k] ROLLOUT FAILED, see $LG"; continue; fi
  grep '^\[rho\]' $LG 2>/dev/null | tail -1
  for spec in "stu:$STU" "$TN:$TEA"; do
    NM=${spec%%:*}; MD=${spec#*:}
    OUT=$R/ent_${NM}_${TAG}_sh${k}.npz
    [ -s $OUT ] || "$E/bin/python" $S/fork_entropy.py --model "$MD" --rollouts $RO \
        --out $OUT --chunk $CH > $L/rho_ent_${NM}_${TAG}_sh${k}.log 2>&1
    [ -s $OUT ] || echo "  [$TAG sh$k] ENTROPY FAILED $NM"
    echo "  [$TAG sh$k] $NM done $(date -Is)"
  done
  echo "==== $TAG cumulative through shard $k ===="
  "$E/bin/python" $S/rho_agg.py --student "$R/ent_stu_${TAG}_sh*.npz" \
      --teacher "$R/ent_${TN}_${TAG}_sh*.npz" --label "$TAG vs $TN"
done
echo "RHO_DONE $TAG $(date -Is)"
