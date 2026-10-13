
#!/bin/bash
# Evaluate a plain HF model (teacher) with the same pipeline as post_run.sh (no merge) (HE+, LCB v6 under the
# official + fixed extractor, MathArena 8 comps). Portable: works on any node
# with /home/USER/opd + a local HF cache holding the base model.
#   bash teacher_eval.sh <tag> <model_id_or_path> <gpu> <port>
#   e.g. bash post_run.sh opd-1.7b-from4b $SCRATCH/checkpoints/opd_code_17b_from4b 3 8030
# tag: hyphenated (bench file names); the matharena config id is tag with - -> _.
set -u
source /home/USER/opd/env.sh none
TAG=$1; CKPT_ROOT=$2; GPU=$3; PORT=$4; BASE=${5:-Qwen/Qwen3-1.7B}
CFG=${TAG//-/_}
V=$VERL_VENV; E=$CODE_EVAL_ENV; R=$OPD_HOME/matharena; O=$OPD_HOME/bench/official
# HF_HUB_OFFLINE deliberately unset: eval needs hub access for datasets on a fresh node
t0=$(date +%s); log () { echo "[post_run $TAG $(( $(date +%s) - t0 ))s] $*"; }

HF_OUT=$CKPT_ROOT   # a plain HF model id / path: no merge

# MathArena model config (port-specific)
sed -e "s#^model:.*#model: $HF_OUT#" -e "s#^max_tokens:.*#max_tokens: ${MAX_TOKENS:-8192}#" -e "s#^concurrent_requests:.*#concurrent_requests: ${CONCURRENCY:-256}#" -e "s#^base_url:.*#base_url: http://localhost:$PORT/v1#" \
    -e "s#^human_readable_id:.*#human_readable_id: $TAG#" -e "s#^date:.*#date: \"$(date +%F)\"#" \
    "$R/configs/models/local/opd_1p7b_from14b.yaml" > "$R/configs/models/local/$CFG.yaml"

log "eval_v5 (HE+, LCB gen, MathArena) on GPU $GPU port $PORT"
bash $OPD_HOME/scripts/eval_v5.sh "$CFG" "$HF_OUT" "$TAG" "$PORT" "$GPU" 1 > $SCRATCH/logs/v5_${TAG}.log 2>&1
tail -1 $SCRATCH/logs/v5_${TAG}.log

[ "${MATH_ONLY:-0}" = 1 ] && { log "MATH_ONLY: skip code scoring"; log "POST_RUN_DONE"; exit 0; }
log "scoring"
export PATH="$E/bin:$PATH" PYTHONPATH=$OPD_HOME/LiveCodeBench
rm -f "$O/humanevalplus_${TAG}_eval_results.json" "$O/lcb_v6_${TAG}"*eval*.json
echo "==== HumanEval+ : $TAG"
"$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$O/humanevalplus_${TAG}.jsonl" 2>&1 | grep -aE "^pass@1|humaneval"
cd $OPD_HOME/LiveCodeBench
"$E/bin/python" $OPD_HOME/scripts/lcb_reextract.py "$TAG"
for T in "$TAG" "${TAG}_fix"; do
  echo "==== LCB v6 : $T"
  "$E/bin/python" -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$O/lcb_v6_$T.json" --release_version release_v6 \
      --start_date 2025-01-01 --scenario codegeneration \
      --num_process_evaluate 16 --timeout 8 > $SCRATCH/logs/lcbs_$T.log 2>&1
  "$E/bin/python" -c "import json;d=json.load(open('$O/lcb_v6_${T}_codegeneration_output_eval.json'));print({k:v for k,v in d[0].items() if 'pass' in k} if isinstance(d,list) else {k:v for k,v in d.items() if 'pass' in k})" 2>/dev/null || grep -aE "pass@1" $SCRATCH/logs/lcbs_$T.log | tail -2
done
log "POST_RUN_DONE"
