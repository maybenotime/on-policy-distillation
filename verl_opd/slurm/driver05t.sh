#!/bin/bash
# q3i17_tpM500_flat: Math-Step500's OWN preferences, with its OWN per-position entropy profile
# replaced by a constant equal to its own sequence-mean entropy (recomputed every step).
# The matched control for M500 teaching directly (+14.0 mean-6 / +13.9 mean-5 at s50): same logits,
# same mean entropy, only the per-position entropy variation removed.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=q3i17_tpM500_flat
STU=Qwen/Qwen3-1.7B
M5=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500

echo "DRIVER05T_START $(date -Is)"
SAVE_FREQ=50 bash $S/chain19_tp.sh $TAG flat $M5 $M5 0,1,2 /tmp/ray_q3flat 1.0 100
echo "TRAIN_RC_Q3M500FLAT=$? $(date -Is)"

export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $STU > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
for ST in 100 50; do
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT $CK — skipping s$ST"; continue; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$STU').save_pretrained('$HF')"
  fi
  echo "MERGE_DONE q3m500flat s$ST $(date -Is)"
  ( ev q3TP_M500_flat-s${ST}-16k  0 8220 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_q3m500flat${ST}_a.out 2>&1 &
  ( ev q3TP_M500_flat-s${ST}b-16k 1 8221 custom/beyondaime ) > $H/ev_q3m500flat${ST}_b.out 2>&1 &
  ( ev q3TP_M500_flat-s${ST}c-16k 2 8222 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_q3m500flat${ST}_c.out 2>&1 &
  wait
done
echo "DRIVER05T_DONE $(date -Is)"
