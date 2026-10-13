#!/bin/bash
# One-off: export HumanEval+ problems to JSON (so the vLLM generation script
# never has to import evalplus -- installing evalplus into the LiveCodeBench
# venv would downgrade protobuf 6.33->5.29 and risk breaking vllm there),
# and confirm the LiveCodeBench v6 problem count.
set -x
export HF_HOME=/localscratch/USER/hf_cache
mkdir -p /home/USER/opd/bench

/home/USER/opd/envs/evalplus_env/bin/python - <<'PY'
import json
from evalplus.data import get_human_eval_plus
p = get_human_eval_plus()
out = {k: {"prompt": v["prompt"], "entry_point": v["entry_point"]} for k, v in p.items()}
json.dump(out, open("/home/USER/opd/bench/humanevalplus_problems.json", "w"))
print("HumanEval+ exported:", len(out))
PY

cd /home/USER/opd/LiveCodeBench
.venv/bin/python - <<'PY'
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
d = load_code_generation_dataset(release_version="release_v6", start_date="2025-01-01")
print("LCB v6 problems (start_date=2025-01-01):", len(d))
print("first question_id:", d[0].question_id)
PY
echo "BENCH_PREP_EXIT=$?"
