#!/bin/bash
# cs-venus-19 job 271952: fill the missing checkpoints of the two direct-teaching runs.
#   GPU 0: q3i17 <- Math-Step500 (DAPO-17k)  s100  (8k)
#   GPU 1: q3i17 <- Math-Step500 (DAPO-17k)  s150  (8k)
#   GPU 2+3: ds15b <- JustRL (DAPO-17k)      s150  (16k), comps split over the two GPUs
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs; C=$SCRATCH/checkpoints
Q3=Qwen/Qwen3-1.7B; DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ALL="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026 custom/amc custom/beyondaime"
merge () {  # fsdp_run step gpu base -> HF dir
  local RUN=$1 ST=$2 G=$3 BASE=$4 HF=$5
  local CK=$C/$RUN/global_step_$ST/actor
  [ -d $CK ] || { echo "NO CHECKPOINT $CK"; return 1; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=$G "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF > $H/merge_$(basename $HF).out 2>&1 || { echo "MERGE_FAILED $HF"; return 1; }
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')" >> $H/merge_$(basename $HF).out 2>&1
  fi
  echo "MERGE_DONE $HF $(date -Is)"
}
ev () { local T=$1 HF=$2 G=$3 P=$4 BASE=$5 MT=$6; shift 6
  for i in $(seq 1 40); do m=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 3000 ] && break; sleep 15; done
  MAX_TOKENS=$MT COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
echo "DRIVER19X_START $(date -Is)"
( merge opd_q3i17_fromK4Bmath500_dapo_16k 100 0 $Q3 $C/hf_q3i17_fromM500_step100 && ev q3i17_fromM500-s100-16k $C/hf_q3i17_fromM500_step100 0 8320 $Q3 8192 $ALL ) > $H/x19_0.out 2>&1 &
( merge opd_q3i17_fromK4Bmath500_dapo_16k 150 1 $Q3 $C/hf_q3i17_fromM500_step150 && ev q3i17_fromM500-s150-16k $C/hf_q3i17_fromM500_step150 1 8321 $Q3 8192 $ALL ) > $H/x19_1.out 2>&1 &
( merge opd_ds15b_fromJustRL_dapo_16k 150 2 $DS $C/hf_ds15b_fromJustRL_step150 && \
  { ev ds15b_fromJustRL-s150-16k  $C/hf_ds15b_fromJustRL_step150 2 8322 $DS 16384 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 > $H/x19_2a.out 2>&1 &
    ev ds15b_fromJustRL-s150b-16k $C/hf_ds15b_fromJustRL_step150 3 8323 $DS 16384 hmmt/hmmt_feb_2026 custom/amc custom/beyondaime > $H/x19_2b.out 2>&1 &
    wait; } ) > $H/x19_2.out 2>&1 &
wait
cat $H/x19_*.out
echo "DRIVER19X_DONE $(date -Is)"
