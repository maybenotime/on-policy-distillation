#!/bin/bash
# Code-only eval (HE+, LCB v6 official+fixed extractor) of one HF model on one GPU.
#   bash teacher_code_eval.sh <tag> <model> <gpu>
# Same generation settings as eval_v5.sh (T=0.7 top_p=0.8 top_k=20 n=4 max_tokens 8192),
# so the numbers are directly comparable with the OPD students. Skips generation if
# outputs exist; always (re)scores, and scores the _fix extraction for base models too.
set -u
source /home/USER/opd/env.sh none
TAG=$1; MODEL=$2; GPU=$3
E=$CODE_EVAL_ENV; O=$OPD_HOME/bench/official
export CUDA_VISIBLE_DEVICES=$GPU PATH="$E/bin:$PATH" PYTHONPATH=$OPD_HOME/LiveCodeBench
SAMP="--temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --n 4 --max_tokens 8192"
t0=$(date +%s); log () { echo "[tce $TAG $(( $(date +%s) - t0 ))s] $*"; }
cd $OPD_HOME/LiveCodeBench
for BENCH in humanevalplus lcb_v6; do
  if [ "$BENCH" = humanevalplus ]; then F="$O/humanevalplus_${TAG}.jsonl"; else F="$O/lcb_v6_${TAG}_raw.json"; fi
  if [ -s "$F" ]; then log "SKIP gen $BENCH"; continue; fi
  log "gen $BENCH"
  "$E/bin/python" $OPD_HOME/scripts/gen2.py --model "$MODEL" --bench "$BENCH" \
      --out_dir "$O" --tag "$TAG" $SAMP --tp 1 --gpu_mem_util 0.85 2>&1 | grep -aE "\[gen\]|rror|it/s\]$" | tail -20
done
log "scoring"
rm -f "$O/humanevalplus_${TAG}_eval_results.json" "$O/lcb_v6_${TAG}"*eval*.json
echo "==== HumanEval+ : $TAG"
"$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$O/humanevalplus_${TAG}.jsonl" 2>&1 | grep -aE "^pass@1|humaneval"
"$E/bin/python" $OPD_HOME/scripts/lcb_reextract.py "$TAG"
for T in "$TAG" "${TAG}_fix"; do
  echo "==== LCB v6 : $T"
  "$E/bin/python" -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$O/lcb_v6_$T.json" --release_version release_v6 \
      --start_date 2025-01-01 --scenario codegeneration \
      --num_process_evaluate 16 --timeout 8 > $SCRATCH/logs/lcbs_$T.log 2>&1
  "$E/bin/python" -c "import json;d=json.load(open('$O/lcb_v6_${T}_codegeneration_output_eval.json'));print({k:v for k,v in d[0].items() if 'pass' in k} if isinstance(d,list) else {k:v for k,v in d.items() if 'pass' in k})" 2>/dev/null || grep -aE "pass@1" $SCRATCH/logs/lcbs_$T.log | tail -2
done
log "TCE_DONE"
