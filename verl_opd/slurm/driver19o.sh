#!/bin/bash
# Resume q3i17_tpM500_shapeA4B (M500's preferences + Qwen3-4B's entropy shape) from its s50
# checkpoint to step 200, then evaluate s200.  The checkpoint lives on THIS node's /localscratch,
# so it has to wait for node 19's GPUs rather than use the ones freed on node 05.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=q3i17_tpM500_shapeA4B
M5=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500      # preferences (A)
Q4=Qwen/Qwen3-4B                                      # entropy shape source (B)
STU=Qwen/Qwen3-1.7B

echo "DRIVER19O_WAIT $(date -Is) — waiting for GPUs 0,1,2 on this node"
ok=0
for _ in $(seq 1 720); do                              # up to 12 h
  free=1
  for i in 0 1 2; do
    u=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ "${u:-99999}" -lt 3000 ] 2>/dev/null || free=0
  done
  if [ $free -eq 1 ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 3 ] && break                               # three consecutive clear checks
  sleep 60
done
[ $ok -ge 3 ] || { echo "GPUS_NEVER_FREED $(date -Is)"; exit 1; }

echo "DRIVER19O_TRAIN_START $(date -Is) (resuming from $(cat $SCRATCH/checkpoints/opd_$TAG/latest_checkpointed_iteration.txt 2>/dev/null))"
SAVE_FREQ=50 bash $S/chain19_tp.sh $TAG shape $M5 $Q4 0,1,2 /tmp/ray_q3m500b 1.0 200
echo "TRAIN_RC_M500A4B=$? $(date -Is)"

export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $STU > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
for ST in 200 100; do
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT $CK — skipping s$ST"; continue; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$STU').save_pretrained('$HF')"
  fi
  echo "MERGE_DONE m500a4b s$ST $(date -Is)"
  ( ev q3TP_M500_shapeA4B-s${ST}-16k  0 8190 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_m500a4b${ST}_a.out 2>&1 &
  ( ev q3TP_M500_shapeA4B-s${ST}b-16k 1 8191 custom/beyondaime ) > $H/ev_m500a4b${ST}_b.out 2>&1 &
  ( ev q3TP_M500_shapeA4B-s${ST}c-16k 2 8192 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_m500a4b${ST}_c.out 2>&1 &
  wait
done
echo "DRIVER19O_DONE $(date -Is)"
