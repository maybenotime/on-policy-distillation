#!/bin/bash
# Phase 1: evaluate both s100 checkpoints (6 GPUs, comps split 3 ways per run).
# Phase 2: resume both runs from s100 to step 200 (verl resume_mode=auto), then their s200/s50 evals.
# The s100 checkpoints live on this node's /localscratch, which node 19 cannot see, so phase 1 must run here.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
JR=hbx/JustRL-DeepSeek-1.5B
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1

merge () {  # tag step gpu -> echoes HF dir
  local TAG=$1 ST=$2 G=$3
  local CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  local HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  if [ ! -s $HF/config.json ]; then
    [ -d $CK ] || { echo "NO CHECKPOINT $CK" >&2; return 1; }
    CUDA_VISIBLE_DEVICES=$G "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF >&2 2>&1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')" >&2 2>&1
  fi
  echo $HF
}
ev () {  # evaltag hfdir gpu port comps...
  local T=$1 HF=$2 G=$3 P=$4; shift 4
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"
}
C_AMC="custom/amc hmmt/hmmt_feb_2026"
C_BEY="custom/beyondaime"
C_AIME="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025"

echo "DRIVER05P_PHASE1_START $(date -Is)"
HF1=$(merge ds15b_tpStu_shapeJustRL 100 0) || exit 1
HF2=$(merge ds15b_tpJustRL_shapeStu 100 3) || exit 1
echo "MERGES_DONE $HF1 $HF2 $(date -Is)"
( ev dsTP_Stu_shapeJustRL-s100-16k  $HF1 0 8120 $C_AMC  ) > $H/ev_s100_1a.out 2>&1 &
( ev dsTP_Stu_shapeJustRL-s100b-16k $HF1 1 8121 $C_BEY  ) > $H/ev_s100_1b.out 2>&1 &
( ev dsTP_Stu_shapeJustRL-s100c-16k $HF1 2 8122 $C_AIME ) > $H/ev_s100_1c.out 2>&1 &
( ev dsTP_JustRL_shapeStu-s100-16k  $HF2 3 8123 $C_AMC  ) > $H/ev_s100_2a.out 2>&1 &
( ev dsTP_JustRL_shapeStu-s100b-16k $HF2 4 8124 $C_BEY  ) > $H/ev_s100_2b.out 2>&1 &
( ev dsTP_JustRL_shapeStu-s100c-16k $HF2 5 8125 $C_AIME ) > $H/ev_s100_2c.out 2>&1 &
wait
echo "DRIVER05P_PHASE1_DONE $(date -Is)"

echo "DRIVER05P_PHASE2_START $(date -Is)"
( SAVE_FREQ=50 bash $S/chain05_tp.sh ds15b_tpStu_shapeJustRL shape   $BASE $JR 0,1,2 /tmp/ray_stu200b 1.0 200
  echo "TRAIN_RC_STU=$? $(date -Is)" ) > $H/resume_stu.out 2>&1 &
( SAVE_FREQ=50 bash $S/chain05_tp.sh ds15b_tpJustRL_shapeStu student $JR   -   3,4,5 /tmp/ray_jrstu2  1.0 200
  echo "TRAIN_RC_JRSTU=$? $(date -Is)" ) > $H/resume_jrstu.out 2>&1 &
wait
echo "DRIVER05P_PHASE2_DONE $(date -Is)"

echo "DRIVER05P_PHASE3_START $(date -Is)"
HF3=$(merge ds15b_tpStu_shapeJustRL 200 0) && {
  ( ev dsTP_Stu_shapeJustRL-s200-16k  $HF3 0 8130 $C_AMC  ) > $H/ev_s200_1a.out 2>&1 &
  ( ev dsTP_Stu_shapeJustRL-s200b-16k $HF3 1 8131 $C_BEY  ) > $H/ev_s200_1b.out 2>&1 &
  ( ev dsTP_Stu_shapeJustRL-s200c-16k $HF3 2 8132 $C_AIME ) > $H/ev_s200_1c.out 2>&1 & }
HF4=$(merge ds15b_tpJustRL_shapeStu 200 3) && {
  ( ev dsTP_JustRL_shapeStu-s200-16k  $HF4 3 8133 $C_AMC  ) > $H/ev_s200_2a.out 2>&1 &
  ( ev dsTP_JustRL_shapeStu-s200b-16k $HF4 4 8134 $C_BEY  ) > $H/ev_s200_2b.out 2>&1 &
  ( ev dsTP_JustRL_shapeStu-s200c-16k $HF4 5 8135 $C_AIME ) > $H/ev_s200_2c.out 2>&1 & }
wait
echo "DRIVER05P_DONE $(date -Is)"
