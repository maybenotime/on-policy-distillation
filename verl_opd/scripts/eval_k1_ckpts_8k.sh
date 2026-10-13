#!/bin/bash
# 19 GPUs 0-2 (training finished): MathArena (5 comps, 8k) evals of the k1+PG Base-student run at steps 271 / 150 / 50, in parallel.
source /home/USER/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1 MAX_MODEL_LEN=32768
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime"
CK=$SCRATCH/checkpoints/opd_q3b17_fromQ3B4Bgrpo_dapo_16k; BASE=Qwen/Qwen3-1.7B-Base
merge () { [ -s $2/config.json ] && return; CUDA_VISIBLE_DEVICES=$3 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK/global_step_$1/actor --target_dir $2 2>&1 | tail -1; "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$2')"; }
te () { bash $S/teacher_eval.sh $1 $2 $3 $4 > $L/teval_$1.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
for s in 271 150 50; do merge $s $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k1_step$s 0; done
( te q3b17-BaseGRPO-k1-s271-8k $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k1_step271 0 8070 ) &
sleep 20; ( te q3b17-BaseGRPO-k1-s150-8k $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k1_step150 1 8071 ) &
sleep 20; ( te q3b17-BaseGRPO-k1-s50-8k $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k1_step50 2 8072 ) &
wait
echo "EVAL_K1_8K_DONE $(date -Is)"
