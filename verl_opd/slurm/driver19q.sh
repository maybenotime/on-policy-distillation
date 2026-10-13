#!/bin/bash
# q3i17_ts_mismatchW: plain k1_probe OPD from Math-Step500, with each token's loss weighted by its rank
# mismatch |rank_s - rank_t| (student vs teacher entropy ordering within the sequence), normalised to mean 1
# per sequence, capped at 5.  200 steps, evaluated at s50 / s100 / s200 against M500 teaching directly
# (+14.0 / — / +16.2 mean-6).
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=q3i17_ts_mismatchW
STU=Qwen/Qwen3-1.7B
M5=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500

echo "DRIVER19Q_START $(date -Is)"
TOKEN_SELECT=mismatch_w TOKEN_WCAP=5 SAVE_FREQ=50 bash $S/chain19_ts.sh $TAG $M5 0,1,2 /tmp/ray_tsw 200
echo "TRAIN_RC_TSW=$? $(date -Is)"

export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $STU > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
for ST in 200 100 50; do
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT $CK — skipping s$ST"; continue; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$STU').save_pretrained('$HF')"
  fi
  echo "MERGE_DONE tsw s$ST $(date -Is)"
  ( ev q3TS_mismatchW-s${ST}-16k  0 8260 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_tsw${ST}_a.out 2>&1 &
  ( ev q3TS_mismatchW-s${ST}b-16k 1 8261 custom/beyondaime ) > $H/ev_tsw${ST}_b.out 2>&1 &
  ( ev q3TS_mismatchW-s${ST}c-16k 2 8262 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_tsw${ST}_c.out 2>&1 &
  wait
done
echo "DRIVER19Q_DONE $(date -Is)"
