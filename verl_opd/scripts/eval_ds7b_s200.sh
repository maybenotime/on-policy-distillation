#!/bin/bash
# 19 GPU 0: MathArena (7 comps, 16k) eval of the DS-7B-distilled student at step 200 (matches the JustRL s200 eval)
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
CK=$SCRATCH/checkpoints/opd_ds15b_fromDS7B_dapo_16k; BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
HF=$SCRATCH/checkpoints/hf_ds15b_fromDS7B_step200
if [ ! -s $HF/config.json ]; then CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK/global_step_200/actor --target_dir $HF 2>&1 | tail -1; "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"; fi
bash $S/teacher_eval.sh ds15b-DS7B-s200-16k $HF 0 8075 > $L/teval_ds15b-DS7B-s200-16k.log 2>&1
echo "EVAL_DONE ds15b-DS7B-s200-16k rc=$? $(date -Is)"
