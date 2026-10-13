#!/bin/bash
# smoke test: 40 problems per model, then grade
source /home/sya225/opd/env.sh none; PY=$CODE_EVAL_ENV/bin/python; S=$OPD_HOME/scripts; export PATH="$CODE_EVAL_ENV/bin:$PATH" TOKENIZERS_PARALLELISM=false
D=/home/sya225/opd/verl_data/deepmath/sample45k.parquet; O=$SCRATCH/deepmath; mkdir -p $O
N=$1; M=$2; G=$3
CUDA_VISIBLE_DEVICES=$G $PY $S/deepmath_gen.py --model $M --data $D --shard 0 --nshards 1 --limit 40 --chunk 40 --out $O/smoke_$N.jsonl 2>&1 | grep -E "^\[gen\]|Error|error" 
$PY $S/deepmath_grade.py --data $D --gen $O/smoke_$N.jsonl --out $O/smoke_grade_$N.jsonl 2>&1 | tail -2
$PY - <<PYE
import json
rs=[json.loads(l) for l in open("$O/smoke_grade_$N.jsonl")]
print("finish:", {k: sum(r["finish_reason"]==k for r in rs) for k in set(r["finish_reason"] for r in rs)})
print("ntok mean %.0f max %d" % (sum(r["ntok"] for r in rs)/len(rs), max(r["ntok"] for r in rs)))
print("preds sample:", [ (r["correct"], r["pred"]) for r in rs[:8] ])
PYE
echo "SMOKE_DONE $N $(date -Is)"
