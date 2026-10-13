#!/bin/bash
# 19 GPU 3: MathArena (5 comps) evals of the k3 Base-student run at 8k max_tokens (Base-family students loop to the cap;
# their real answers are 1-2.5k words). Fresh tags (-8k) so all checkpoints share the same setting. Order: s150, s200, s50, base.
source /home/USER/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1 MAX_MODEL_LEN=32768
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime"
CK=$SCRATCH/checkpoints/opd_q3b17_fromQ3B4Bgrpo_dapo_16k_k3; BASE=Qwen/Qwen3-1.7B-Base
merge () { [ -s $2/config.json ] && return; CUDA_VISIBLE_DEVICES=3 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK/global_step_$1/actor --target_dir $2 2>&1 | tail -1; "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$2')"; }
te () { bash $S/teacher_eval.sh $1 $2 3 8068 > $L/teval_$1.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
for s in 150 200 50; do merge $s $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k3_step$s; te q3b17-BaseGRPO-k3-s$s-8k $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k3_step$s; done
te Q3-1.7B-Base-8k $BASE
echo "EVAL_K3_8K_DONE $(date -Is)"
