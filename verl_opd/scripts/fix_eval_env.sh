#!/bin/bash
# Two fixes to the throwaway code_eval_env:
#
# 1) anthropic >=0.30 removed the legacy HUMAN_PROMPT / AI_PROMPT constants, but
#    lcb_runner/prompts/code_generation.py still does
#      `from anthropic import HUMAN_PROMPT, AI_PROMPT`
#    which breaks the whole scoring import chain (scenario_router ->
#    custom_evaluator). We never call the Anthropic API here -- the constants are
#    only needed to satisfy the import -- so append their historical values.
#    Scoped to this eval-only venv; nothing else on the box is touched.
#
# 2) vllm's compile backend shells out to `ninja`; the wheel puts the binary in
#    the venv's bin/, which is not on PATH unless we export it.
set -x
E=/home/sya225/opd/envs/code_eval_env
A="$E/lib/python3.12/site-packages/anthropic/__init__.py"

grep -q "HUMAN_PROMPT" "$A" || cat >> "$A" <<'PY'

# --- added for LiveCodeBench compatibility (eval-only venv) ---
# lcb_runner still imports these legacy constants, dropped in anthropic>=0.30.
HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"
PY

ls -l "$E/bin/ninja"
export PATH="$E/bin:$PATH"

echo "===== re-verify ====="
PYTHONPATH=/home/sya225/opd/LiveCodeBench "$E/bin/python" -c "
import shutil
from anthropic import HUMAN_PROMPT, AI_PROMPT
from lcb_runner.runner.scenario_router import build_prompt_benchmark, get_metrics
from lcb_runner.prompts.code_generation import PromptConstants
from lcb_runner.utils.extraction_utils import extract_code
print('scenario_router + prompts + extraction: OK')
print('ninja at', shutil.which('ninja'))
"
echo "FIX_DONE=$?"
