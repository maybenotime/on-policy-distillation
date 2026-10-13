#!/bin/bash
# 19 GPU 3: MathArena-only eval (16k) of the JustRL-distilled R1-Distill-1.5B student at step 200 (last healthy) and
# step 250 (after the entropy blow-up), plus the untrained base model for reference.
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
CK=$SCRATCH/checkpoints/opd_ds15b_fromJustRL_dapo_16k; BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
for s in 200 250; do
  HF=$SCRATCH/checkpoints/hf_ds15b_fromJustRL_step$s
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=3 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK/global_step_$s/actor --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
  fi
done
te () { bash $S/teacher_eval.sh $1 $2 3 8060 > $L/teval_$1.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
te ds15b-JustRL-s200-16k $SCRATCH/checkpoints/hf_ds15b_fromJustRL_step200
te ds15b-JustRL-s250-16k $SCRATCH/checkpoints/hf_ds15b_fromJustRL_step250
te DS15B-base-16k $BASE
echo "EVAL_DS_CKPTS_DONE $(date -Is)"
