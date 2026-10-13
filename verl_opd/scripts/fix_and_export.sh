#!/bin/bash
# 1) vLLM's compile backend shells out to `ninja`; it is not on PATH in the
#    verl_opd venv, which killed engine startup with FileNotFoundError: 'ninja'.
# 2) The LCB v6 arrow cache was built by datasets 2.21 (LiveCodeBench venv).
#    verl_opd has datasets 4.x, which cannot read that cache layout
#    ("Couldn't find cache for livecodebench/code_generation_lite").
#    So export the raw questions from the LCB venv once; generation then needs
#    no lcb_runner import and no datasets-version agreement at all.
set -x
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/USER/uv_cache
export HF_HOME=/localscratch/USER/hf_cache
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets

uv pip install --python /home/USER/opd/envs/verl_opd/bin/python ninja 2>&1 | tail -3
/home/USER/opd/envs/verl_opd/bin/python -c "import shutil;print('ninja at', shutil.which('ninja'))"
ls -l /home/USER/opd/envs/verl_opd/bin/ninja

cd /home/USER/opd/LiveCodeBench
PYTHONPATH=/home/USER/opd/LiveCodeBench .venv/bin/python - <<'PY'
import json
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
d = load_code_generation_dataset(release_version="release_v6", start_date="2025-01-01")
out = [{"question_id": str(q.question_id),
        "question_content": q.question_content,
        "starter_code": q.starter_code or ""} for q in d]
json.dump(out, open("/home/USER/opd/bench/lcb_v6_questions.json", "w"))
print("LCB v6 questions exported:", len(out))
PY
echo "FIX_EXPORT_DONE=$?"
