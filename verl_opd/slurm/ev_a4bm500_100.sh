#!/bin/bash
# Fill the missing s100 point of q3i17_tpA4B_shapeM500 (4B prefs + M500 shape) on the idle GPU 3 of node 19,
# so its curve (s50/s100/s200) matches the mirror run M500 prefs + 4B shape.  Single GPU, all 7 competitions.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs
TAG=q3i17_tpA4B_shapeM500; STU=Qwen/Qwen3-1.7B
CK=$SCRATCH/checkpoints/opd_$TAG/global_step_100/actor
HF=$SCRATCH/checkpoints/hf_${TAG}_step100
[ -d $CK ] || { echo "NO CHECKPOINT $CK"; exit 1; }
if [ ! -s $HF/config.json ]; then
  CUDA_VISIBLE_DEVICES=3 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$STU').save_pretrained('$HF')"
fi
echo "MERGE_DONE a4bm500 s100 $(date -Is)"
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
bash $S/teacher_eval.sh q3TP_A4B_shapeM500-s100-16k $HF 3 8270 $STU > $L/teval_a4bm500_100.log 2>&1
echo "EVAL_DONE a4bm500-s100 rc=$? $(date -Is)"
