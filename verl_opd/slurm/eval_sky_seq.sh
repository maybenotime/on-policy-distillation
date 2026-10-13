#!/bin/bash
# 19 GPU 0: sequential MathArena (7 comps, 16k) eval of the Skywork-OR1-Math-7B-distilled DS-1.5B student, s200 then s50.
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
CK=$SCRATCH/checkpoints/opd_ds15b_fromSky7B_dapo_16k; BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
for ST in 200 50; do
  HF=$SCRATCH/checkpoints/hf_ds15b_fromSky7B_step$ST
  if [ ! -s $HF/config.json ]; then
    echo "MERGE step $ST $(date -Is)"
    CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp \
        --local_dir $CK/global_step_$ST/actor --target_dir $HF 2>&1 | tail -2
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
  fi
  T=ds15b-Sky7B-s$ST-16k
  echo "EVAL_START $T $(date -Is)"
  bash $S/teacher_eval.sh $T $HF 0 8077 > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"
done
echo "ALL_DONE $(date -Is)"
