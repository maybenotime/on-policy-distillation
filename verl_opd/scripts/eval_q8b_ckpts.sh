#!/bin/bash
# 19: MathArena (7 comps, 8k, n=4) evals of the Qwen3-1.7B <- 8B student at steps 200 (GPU 0) and 50 (GPU 1), in parallel.
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
CK=$SCRATCH/checkpoints/opd_q3i17_from8B_dapo_16k; BASE=Qwen/Qwen3-1.7B
merge () { [ -s $2/config.json ] && return; CUDA_VISIBLE_DEVICES=$3 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK/global_step_$1/actor --target_dir $2 2>&1 | tail -1; "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$2')"; }
te () { bash $S/teacher_eval.sh $1 $2 $3 $4 > $L/teval_$1.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
merge 200 $SCRATCH/checkpoints/hf_q3i17_from8B_step200 0; merge 50 $SCRATCH/checkpoints/hf_q3i17_from8B_step50 1
( te q3i17-8B-s200-8k $SCRATCH/checkpoints/hf_q3i17_from8B_step200 0 8076 ) &
sleep 20; ( te q3i17-8B-s50-8k $SCRATCH/checkpoints/hf_q3i17_from8B_step50 1 8077 ) &
wait; echo "EVAL_Q8B_DONE $(date -Is)"
