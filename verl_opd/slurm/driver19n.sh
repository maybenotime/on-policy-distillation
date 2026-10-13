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

echo "DRIVER19N_WAIT_FOR_JR100 $(date -Is)"
for _ in $(seq 1 240); do            # up to 4 h
  grep -q JR100_ALL_DONE $H/ev_jr100.out 2>/dev/null && break
  sleep 60
done
for _ in $(seq 1 60); do             # wait for the GPUs to actually come back
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${u:-99999}" -lt 3000 ] 2>/dev/null && break
  sleep 20
done
echo "DRIVER19N_TRAIN_START $(date -Is)"
SAVE_FREQ=50 bash $S/chain05_tp.sh $TAG flat $BASE $JR 0,1,2 /tmp/ray_flat 1.0 50
echo "TRAIN_RC_FLAT=$? $(date -Is)"

CK=$SCRATCH/checkpoints/opd_$TAG/global_step_50/actor
HF=$SCRATCH/checkpoints/hf_${TAG}_step50
if [ ! -s $HF/config.json ]; then
  [ -d $CK ] || { echo "NO CHECKPOINT $CK"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
      --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
echo "MERGE_DONE flat s50 $(date -Is)"
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
( ev dsTP_Stu_flat-s50-16k  0 8150 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_flat_a.out 2>&1 &
( ev dsTP_Stu_flat-s50b-16k 1 8151 custom/beyondaime ) > $H/ev_flat_b.out 2>&1 &
( ev dsTP_Stu_flat-s50c-16k 2 8152 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_flat_c.out 2>&1 &
wait
echo "DRIVER19N_DONE $(date -Is)"
