#!/bin/bash
# Reorder node 19: as soon as flat's TRAINING finishes, stop its driver before it grabs GPUs 0,1,2
# for evaluation, so the queued M500-prefs+4B-shape resume starts immediately.  flat's s200/s100
# evals are then run here, after that resume is done.  (flat's s50 eval is already running on GPU 3
# and is left alone.)
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=ds15b_tpStu_flat; BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

echo "DRIVER19P_WAIT_FLAT_TRAIN $(date -Is)"
for _ in $(seq 1 900); do
  grep -q TRAIN_RC_FLAT $H/driver19n2.out 2>/dev/null && break
  sleep 60
done
grep -q TRAIN_RC_FLAT $H/driver19n2.out 2>/dev/null || { echo "FLAT_TRAIN_NEVER_ENDED $(date -Is)"; exit 1; }
echo "FLAT_TRAIN_DONE — stopping its driver so its evals do not hold GPUs 0,1,2 $(date -Is)"
tmux kill-session -t tp_flat 2>/dev/null
sleep 5
for p in $(pgrep -f ray_flat); do kill -9 $p 2>/dev/null; done
sleep 5
for i in 0 1 2; do for p in $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; done
sleep 10
echo "GPUS_RELEASED $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ') $(date -Is)"

echo "DRIVER19P_WAIT_M500A4B $(date -Is)"
for _ in $(seq 1 1440); do
  grep -q DRIVER19O_DONE $H/driver19o.out 2>/dev/null && break
  sleep 60
done
echo "M500A4B_DONE_OR_TIMEOUT $(date -Is) — now running flat's deferred evals"

export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
for ST in 200 100; do
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT $CK — skipping s$ST"; continue; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
  fi
  echo "MERGE_DONE flat s$ST $(date -Is)"
  ( ev dsTP_Stu_flat-s${ST}-16k  0 8200 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_flat${ST}_a.out 2>&1 &
  ( ev dsTP_Stu_flat-s${ST}b-16k 1 8201 custom/beyondaime ) > $H/ev_flat${ST}_b.out 2>&1 &
  ( ev dsTP_Stu_flat-s${ST}c-16k 2 8202 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_flat${ST}_c.out 2>&1 &
  wait
done
echo "DRIVER19P_DONE $(date -Is)"
