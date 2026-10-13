#!/bin/bash
# Score the already-generated HE+ / LCB outputs of the Code-300 16k student (no GPU).
source /home/USER/opd/env.sh none
TAG=opd-1.7b-fromK4Bcode300-16k; E=$CODE_EVAL_ENV; O=$OPD_HOME/bench/official
export PATH="$E/bin:$PATH" PYTHONPATH=$OPD_HOME/LiveCodeBench
rm -f "$O/humanevalplus_${TAG}_eval_results.json" "$O/lcb_v6_${TAG}"*eval*.json
echo "==== HumanEval+ : $TAG"
"$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$O/humanevalplus_${TAG}.jsonl" 2>&1 | grep -aE "^pass@1|humaneval"
cd $OPD_HOME/LiveCodeBench
"$E/bin/python" $OPD_HOME/scripts/lcb_reextract.py "$TAG"
for T in "$TAG" "${TAG}_fix"; do
  echo "==== LCB v6 : $T"
  "$E/bin/python" -m lcb_runner.runner.custom_evaluator --custom_output_file "$O/lcb_v6_$T.json" --release_version release_v6 \
      --start_date 2025-01-01 --scenario codegeneration --num_process_evaluate 16 --timeout 8 > $SCRATCH/logs/lcbs_$T.log 2>&1
  "$E/bin/python" -c "import json;d=json.load(open('$O/lcb_v6_${T}_codegeneration_output_eval.json'));print({k:v for k,v in d[0].items() if 'pass' in k} if isinstance(d,list) else {k:v for k,v in d.items() if 'pass' in k})" 2>/dev/null || grep -aE "pass@1" $SCRATCH/logs/lcbs_$T.log | tail -2
done
echo SCORE300_DONE
