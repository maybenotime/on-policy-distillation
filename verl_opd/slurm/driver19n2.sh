#!/bin/bash
# ds15b_tpStu_flat: the matched control for tpStu_shapeJustRL.  Preference source = the student's own
# base (so the preference channel is switched off, exactly as in tpStu_shapeJustRL); target entropy =
# the MEAN of JustRL's entropy on the same rollout, recomputed every step (mode=flat).  Difference from
# tpStu_shapeJustRL is therefore only the within-sequence entropy profile.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=ds15b_tpStu_flat
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
JR=hbx/JustRL-DeepSeek-1.5B

echo "DRIVER19N2_START (200 steps) $(date -Is)"
echo "DRIVER19N_TRAIN_START $(date -Is)"
SAVE_FREQ=50 bash $S/chain05_tp.sh $TAG flat $BASE $JR 0,1,2 /tmp/ray_flat 1.0 200
echo "TRAIN_RC_FLAT=$? $(date -Is)"

export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
for ST in 200 100 50; do
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT $CK — skipping s$ST"; continue; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
  fi
  echo "MERGE_DONE flat s$ST $(date -Is)"
  ( ev dsTP_Stu_flat-s${ST}-16k  0 81$((50+ST/50)) custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_flat${ST}_a.out 2>&1 &
  ( ev dsTP_Stu_flat-s${ST}b-16k 1 82$((50+ST/50)) custom/beyondaime ) > $H/ev_flat${ST}_b.out 2>&1 &
  ( ev dsTP_Stu_flat-s${ST}c-16k 2 83$((50+ST/50)) aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_flat${ST}_c.out 2>&1 &
  wait
done
echo "DRIVER19N_DONE $(date -Is)"
