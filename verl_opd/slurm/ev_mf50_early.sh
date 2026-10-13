#!/bin/bash
# Evaluate q3i17_tpM500_flat at s50 NOW on the idle GPUs 3,4,5, so it is off the critical path before
# node 05 expires (~00:37).  Same tags as driver05u's own s50 pass, which will then just reuse the results.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=q3i17_tpM500_flat; STU=Qwen/Qwen3-1.7B
CK=$SCRATCH/checkpoints/opd_$TAG/global_step_50/actor
HF=$SCRATCH/checkpoints/hf_${TAG}_step50
ls $CK/optim_world_size_2_rank_0.pt >/dev/null 2>&1 || { echo "S50_INCOMPLETE_OR_MISSING $(ls $CK 2>&1 | tr '\n' ' ')"; exit 1; }
for i in 3 4 5; do u=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits); [ "${u:-99999}" -lt 3000 ] || { echo "GPU$i_BUSY ($u MiB)"; exit 1; }; done
if [ ! -s $HF/config.json ]; then
  CUDA_VISIBLE_DEVICES=3 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$STU').save_pretrained('$HF')"
fi
echo "MERGE_DONE mf50-early $(date -Is) disk=$(df -h /localscratch | tail -1 | awk '{print $4}')"
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $STU > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
( ev q3TP_M500_flat-s50-16k  3 8280 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_mf50e_a.out 2>&1 &
( ev q3TP_M500_flat-s50b-16k 4 8281 custom/beyondaime ) > $H/ev_mf50e_b.out 2>&1 &
( ev q3TP_M500_flat-s50c-16k 5 8282 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_mf50e_c.out 2>&1 &
wait
echo "MF50_EARLY_DONE $(date -Is)"
