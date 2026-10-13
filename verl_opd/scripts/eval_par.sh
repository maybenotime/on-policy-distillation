#!/bin/bash
# 19, all 4 GPUs: MathArena-only (16k) evals in parallel
#  GPU0: JustRL-student step 200   GPU1: JustRL-student step 250   GPU2: R1-Distill-1.5B base
#  GPU3: Qwen3-1.7B-Base <- 4B-Base-GRPO (k3) step 50, then untrained Qwen3-1.7B-Base for reference
source /home/USER/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"   # math500 dropped, HMMT last
merge () { # ckpt_root step out base
  [ -s $3/config.json ] && return
  CUDA_VISIBLE_DEVICES=3 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $1/global_step_$2/actor --target_dir $3 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$4').save_pretrained('$3')"
}
merge $SCRATCH/checkpoints/opd_ds15b_fromJustRL_dapo_16k 200 $SCRATCH/checkpoints/hf_ds15b_fromJustRL_step200 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
merge $SCRATCH/checkpoints/opd_ds15b_fromJustRL_dapo_16k 250 $SCRATCH/checkpoints/hf_ds15b_fromJustRL_step250 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
merge $SCRATCH/checkpoints/opd_q3b17_fromQ3B4Bgrpo_dapo_16k_k3 50 $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k3_step50 Qwen/Qwen3-1.7B-Base
te () { bash $S/teacher_eval.sh $1 $2 $3 $4 > $L/teval_$1.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
( te ds15b-JustRL-s200-16k $SCRATCH/checkpoints/hf_ds15b_fromJustRL_step200 0 8060 ) &
sleep 20; ( te ds15b-JustRL-s250-16k $SCRATCH/checkpoints/hf_ds15b_fromJustRL_step250 1 8061 ) &
sleep 20; ( te DS15B-base-16k deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 2 8062 ) &
sleep 20; ( te q3b17-BaseGRPO-k3-s50-16k $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k3_step50 3 8063; te Q3-1.7B-Base-16k Qwen/Qwen3-1.7B-Base 3 8063 ) &
wait
echo "EVAL_PAR_DONE $(date -Is)"
