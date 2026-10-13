#!/bin/bash
source /home/sya225/opd/env.sh none; PY=$CODE_EVAL_ENV/bin/python; S=$OPD_HOME/scripts; export PATH="$CODE_EVAL_ENV/bin:$PATH" TOKENIZERS_PARALLELISM=false
D=/home/sya225/opd/verl_data/deepmath/sample45k.parquet; O=$SCRATCH/deepmath; mkdir -p $O
CUDA_VISIBLE_DEVICES=2 $PY $S/deepmath_gen.py --model Qwen/Qwen3-8B --family qwen3nt --max_tokens 16384 --data $D --limit 20 --chunk 20 --out $O/smoke_Q3_8B.jsonl 2>&1 | grep -E "^\[gen\]|Error|error"
$PY $S/deepmath_grade.py --data $D --gen $O/smoke_Q3_8B.jsonl --out $O/smoke_grade_Q3_8B.jsonl 2>&1 | tail -1
$PY - <<PYE
import json
rs=[json.loads(l) for l in open("$O/smoke_grade_Q3_8B.jsonl")]
gs=[json.loads(l) for l in open("$O/smoke_Q3_8B.jsonl")]
print("finish:", {k: sum(r["finish_reason"]==k for r in rs) for k in set(r["finish_reason"] for r in rs)})
print("ntok mean %.0f max %d" % (sum(r["ntok"] for r in rs)/len(rs), max(r["ntok"] for r in rs)))
print("has <think>:", sum("<think>" in g["response"][:50] for g in gs), "/", len(gs))
print("preds:", [(r["correct"], r["pred"][:25]) for r in rs[:8]])
PYE
echo "SMOKE_DONE Q3_8B $(date -Is)"
