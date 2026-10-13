#!/bin/bash
# 19 GPU 3 (training on 0-2): MathArena evals of the k3 Base-student run at key steps, sequential.
# Order: 271 (final) -> 200 (after the rho jump) -> 150 (plateau, before the jump) -> 50 (finish) -> Qwen3-1.7B-Base (finish).
# 5 comps (no math500, no HMMT: Base-family students score ~0 there and loop to 16k); matharena skips finished problems.
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1 MAX_MODEL_LEN=32768
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime"
CK=$SCRATCH/checkpoints/opd_q3b17_fromQ3B4Bgrpo_dapo_16k_k3; BASE=Qwen/Qwen3-1.7B-Base
merge () { [ -s $2/config.json ] && return; CUDA_VISIBLE_DEVICES=3 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK/global_step_$1/actor --target_dir $2 2>&1 | tail -1; "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$2')"; }
te () { bash $S/teacher_eval.sh $1 $2 3 8067 > $L/teval_$1.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
for s in 271 200 150; do merge $s $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k3_step$s; te q3b17-BaseGRPO-k3-s$s-16k $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k3_step$s; done
te q3b17-BaseGRPO-k3-s50-16k $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k3_step50
te Q3-1.7B-Base-16k $BASE
echo "EVAL_K3_CKPTS_DONE $(date -Is)"
