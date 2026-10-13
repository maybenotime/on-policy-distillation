#!/bin/bash
# bash eval19.sh <TAG> <STEP> <GPU> <PORT> <NAME> <BASE_MODEL> <MAX_TOKENS>  — merge + MathArena (7 comps)
# 9-18 20:25: s50 evaluations cancelled by the user; only s100 (and any other explicitly requested step) runs.
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=$7 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
TAG=$1; ST=$2; G=$3; PORT=$4; NAME=$5; BASE=$6; CK=$SCRATCH/checkpoints/opd_$TAG; HF=$SCRATCH/checkpoints/hf_${NAME}_step$ST
if [ ! -s $HF/config.json ]; then
  CUDA_VISIBLE_DEVICES=$G "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK/global_step_$ST/actor --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
T=$NAME-s$ST-${7%k*}; [ "$7" = 16384 ] && T=$NAME-s$ST-16k || T=$NAME-s$ST-8k
echo "EVAL_START $T gpu=$G $(date -Is)"
bash $S/teacher_eval.sh $T $HF $G $PORT > $L/teval_$T.log 2>&1
echo "EVAL_DONE $T rc=$? $(date -Is)"
"$OPD_HOME/matharena/.venv/bin/python" $S/agg_partial.py ${T//-/_} 2>/dev/null | tail -1
