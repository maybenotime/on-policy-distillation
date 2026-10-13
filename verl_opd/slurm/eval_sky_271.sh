#!/bin/bash
# 19 GPU 1: MathArena (7 comps, 16k) eval of the Skywork-distilled DS-1.5B student at the final step 271.
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
CK=$SCRATCH/checkpoints/opd_ds15b_fromSky7B_dapo_16k; BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
HF=$SCRATCH/checkpoints/hf_ds15b_fromSky7B_step271
if [ ! -s $HF/config.json ]; then
  echo "MERGE step 271 $(date -Is)"
  CUDA_VISIBLE_DEVICES=1 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp \
      --local_dir $CK/global_step_271/actor --target_dir $HF 2>&1 | tail -2
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
T=ds15b-Sky7B-s271-16k
echo "EVAL_START $T $(date -Is)"
bash $S/teacher_eval.sh $T $HF 1 8078 > $L/teval_$T.log 2>&1
echo "EVAL_DONE $T rc=$? $(date -Is)"
