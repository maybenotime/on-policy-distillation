#!/bin/bash
# Two jobs on the 13 h this node has left:
#   GPUs 3,4,5 : evaluate ds15b_tpJustRL_shapeStu at s150 (its s200 was lost to the full disk)
#   GPUs 0,1,2 : resume q3i17_tpM500_flat from s50 to 100, then evaluate s100
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
STU=Qwen/Qwen3-1.7B
M5=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
merge () {  # run step gpu base -> prints HF dir on stdout
  local TAG=$1 ST=$2 G=$3 BASE=$4
  local CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  local HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT $CK" >&2; return 1; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=$G "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF >&2 2>&1 || return 1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')" >&2 2>&1
  fi
  echo $HF
}
ev () { local T=$1 HF=$2 G=$3 P=$4 BASE=$5; shift 5
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
C_AMC="custom/amc hmmt/hmmt_feb_2026"; C_BEY="custom/beyondaime"
C_AIME="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025"

echo "DRIVER05U_START $(date -Is)"
(
  export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
  HF=$(merge ds15b_tpJustRL_shapeStu 150 3 $DS) || exit 1
  echo "MERGE_DONE jrstu s150 $(date -Is)"
  ( ev dsTP_JustRL_shapeStu-s150-16k  $HF 3 8230 $DS $C_AMC  ) > $H/ev_jr150_a.out 2>&1 &
  ( ev dsTP_JustRL_shapeStu-s150b-16k $HF 4 8231 $DS $C_BEY  ) > $H/ev_jr150_b.out 2>&1 &
  ( ev dsTP_JustRL_shapeStu-s150c-16k $HF 5 8232 $DS $C_AIME ) > $H/ev_jr150_c.out 2>&1 &
  wait
  echo "JRSTU150_DONE $(date -Is)"
) > $H/jr150.out 2>&1 &

(
  SAVE_FREQ=50 bash $S/chain19_tp.sh q3i17_tpM500_flat flat $M5 $M5 0,1,2 /tmp/ray_q3flat2 1.0 100
  echo "TRAIN_RC_Q3M500FLAT=$? $(date -Is)"
  export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
  for ST in 100 50; do
    HF=$(merge q3i17_tpM500_flat $ST 0 $STU) || continue
    echo "MERGE_DONE q3m500flat s$ST $(date -Is)"
    ( ev q3TP_M500_flat-s${ST}-16k  $HF 0 8240 $STU $C_AMC  ) > $H/ev_q3mf${ST}_a.out 2>&1 &
    ( ev q3TP_M500_flat-s${ST}b-16k $HF 1 8241 $STU $C_BEY  ) > $H/ev_q3mf${ST}_b.out 2>&1 &
    ( ev q3TP_M500_flat-s${ST}c-16k $HF 2 8242 $STU $C_AIME ) > $H/ev_q3mf${ST}_c.out 2>&1 &
    wait
  done
  echo "Q3M500FLAT_DONE $(date -Is)"
) > $H/q3mf.out 2>&1 &
wait
echo "DRIVER05U_DONE $(date -Is)"
