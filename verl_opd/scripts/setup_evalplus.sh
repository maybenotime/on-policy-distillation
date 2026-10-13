#!/bin/bash
# Install evalplus (HumanEval+) into its own venv so it cannot perturb the
# verl_opd training env (evalplus pins its own transformers/vllm range).
set -x
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/localscratch/sya225/uv_cache
export HF_HOME=/localscratch/sya225/hf_cache
E=/home/sya225/opd/envs/evalplus_env
uv venv "$E" --python 3.12 || exit 1
uv pip install --python "$E/bin/python" evalplus 2>&1 | tail -5
"$E/bin/python" -c "
import evalplus, evalplus.data as D
print('evalplus', getattr(evalplus,'__version__','?'))
probs = D.get_human_eval_plus()
print('HumanEval+ problems:', len(probs))
k = list(probs)[0]
print('sample task_id:', k)
"
echo "EVALPLUS_SETUP_EXIT=$?"
