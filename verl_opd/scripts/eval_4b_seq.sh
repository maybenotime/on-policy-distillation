#!/bin/bash
# 05 GPU 0 only: MathArena (7 comps, 8k, n=4) evals of the Qwen3-1.7B <- Math-Step500 student, step 200 then step 50.
true
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
CK=$SCRATCH/checkpoints/opd_q3i17_from4B_dapo_16k; BASE=Qwen/Qwen3-1.7B
merge () { [ -s $2/config.json ] && return; CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK/global_step_$1/actor --target_dir $2 2>&1 | tail -1; "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$2')"; }
te () { bash $S/teacher_eval.sh $1 $2 0 8079 > $L/teval_$1.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
merge 200 $SCRATCH/checkpoints/hf_q3i17_from4B_step200; merge 50 $SCRATCH/checkpoints/hf_q3i17_from4B_step50
te q3i17-4B-s200-8k $SCRATCH/checkpoints/hf_q3i17_from4B_step200
te q3i17-4B-s50-8k $SCRATCH/checkpoints/hf_q3i17_from4B_step50
echo "EVAL_4B_DONE $(date -Is)"
