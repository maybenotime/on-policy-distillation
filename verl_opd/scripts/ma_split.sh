#!/bin/bash
# 19: once LCB gen is done, replace the sequential MathArena stage with two vLLM servers
# (GPU1 full, GPU3 shared with the training teacher) and run scoring concurrently.
source /home/USER/opd/env.sh none
TAG=opd-1.7b-fromRLCR4B; CFG=opd_1.7b_fromRLCR4B; HF=/localscratch/USER/checkpoints/hf_${CFG}_step424
E=$CODE_EVAL_ENV; R=$OPD_HOME/matharena; O=$OPD_HOME/bench/official; L=/localscratch/USER/logs
export PATH="$E/bin:$PATH" VLLM_API_KEY=dummy HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
t0=$(date +%s); log () { echo "[ma_split $(( $(date +%s) - t0 ))s] $*"; }

until [ -s $O/lcb_v6_${TAG}_raw.json ] && grep -q "serving :8031" $L/v5_${TAG}.log; do sleep 20; done
log "LCB gen done; taking over MathArena"
pkill -f "post_run.sh $TAG"; pkill -f "eval_v5.sh $CFG"; pkill -f "api_server.*--port 8031"; sleep 8

serve () {  # gpu port util
  CUDA_VISIBLE_DEVICES=$1 "$E/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$HF" --served-model-name "$HF" --port $2 --dtype bfloat16 --gpu-memory-utilization $3 \
    --max-model-len 40960 --api-key dummy --no-enforce-eager --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    > $L/vllm_${CFG}_$2.log 2>&1 &
  echo $!
}
PA=$(serve 1 8031 0.85); PB=$(serve 3 8032 0.40)

# scoring (CPU) in parallel with serving
(
  export PYTHONPATH=$OPD_HOME/LiveCodeBench
  rm -f "$O/humanevalplus_${TAG}_eval_results.json" "$O/lcb_v6_${TAG}"*eval*.json
  echo "==== HumanEval+ : $TAG"
  "$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$O/humanevalplus_${TAG}.jsonl" 2>&1 | grep -aE "^pass@1|humaneval"
  cd $OPD_HOME/LiveCodeBench
  "$E/bin/python" $OPD_HOME/scripts/lcb_reextract.py "$TAG"
  for T in "$TAG" "${TAG}_fix"; do
    echo "==== LCB v6 : $T"
    "$E/bin/python" -m lcb_runner.runner.custom_evaluator --custom_output_file "$O/lcb_v6_$T.json" \
      --release_version release_v6 --start_date 2025-01-01 --scenario codegeneration \
      --num_process_evaluate 16 --timeout 8 > $L/lcbs_$T.log 2>&1
    "$E/bin/python" -c "import json;d=json.load(open('$O/lcb_v6_${T}_codegeneration_output_eval.json'));print({k:v for k,v in d[0].items() if 'pass' in k} if isinstance(d,list) else {k:v for k,v in d.items() if 'pass' in k})" 2>/dev/null || grep -aE "pass@1" $L/lcbs_$T.log | tail -2
  done
  echo SCORING_DONE
) > $L/score_${TAG}.log 2>&1 &
PS=$!

for p in 8031 8032; do
  for i in $(seq 1 150); do curl -s -H "Authorization: Bearer dummy" "http://localhost:$p/v1/models" >/dev/null 2>&1 && break; sleep 10; done
  log "server :$p up"
done
sed -e "s#^base_url:.*#base_url: http://localhost:8032/v1#" "$R/configs/models/local/$CFG.yaml" > "$R/configs/models/local/${CFG}_b.yaml"
cd "$R"
log "MathArena start (A: math500 on :8031 | B: 7 comps on :8032)"
"$R/.venv/bin/python" scripts/run.py --comp custom/math500 --models "local/$CFG" --n 4 > $L/ma_${TAG}_a.log 2>&1 &
MA=$!
"$R/.venv/bin/python" scripts/run.py --comp aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026 custom/amc custom/beyondaime \
    --models "local/${CFG}_b" --n 4 > $L/ma_${TAG}_b.log 2>&1 &
MB=$!
wait $MA; log "A done"; wait $MB; log "B done"
for d in $R/outputs/*/*/local/${CFG}_b; do [ -d "$d" ] && mv "$d" "${d%_b}"; done
rm -f "$R/configs/models/local/${CFG}_b.yaml"
kill $PA $PB 2>/dev/null
wait $PS; cat $L/score_${TAG}.log
log "MA_SPLIT_DONE"
