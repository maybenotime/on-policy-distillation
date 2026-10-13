#!/bin/bash
# Install the official MathArena harness (eth-sri/matharena) for the math evals.
#
# Local models use `api: custom` + base_url, i.e. we serve the model ourselves
# with vLLM's OpenAI-compatible server and MathArena talks to it over HTTP. That
# means we do NOT need matharena's own vllm extra -- we reuse code_eval_env's
# vllm 0.27.1 for serving, keeping matharena's deps light and isolated.
#
# Its own venv: matharena pulls anthropic/google-genai/lean-explore etc., which
# have no business anywhere near the training or code-eval environments.
set -x
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/localscratch/USER/uv_cache
R=/home/USER/opd/matharena

rm -rf "$R"
git clone --depth 1 https://github.com/eth-sri/matharena.git "$R" 2>&1 | tail -3
cd "$R"

# Interpreter: the uv-managed CPython on shared /home (system python3.12 has no
# headers, which breaks any package needing to build).
uv venv .venv --python 3.12 2>&1 | tail -2
uv pip install --python "$R/.venv/bin/python" -e . 2>&1 | tail -6
echo "INSTALL_RC=$?"

"$R/.venv/bin/python" -c "
import matharena, sys
print('matharena imported OK, python', sys.version.split()[0])" 2>&1 | tail -3

echo "=== run.py 参数 ==="
cd "$R" && "$R/.venv/bin/python" scripts/run.py --help 2>&1 | head -30
echo "SETUP_DONE $(date -Is)"
