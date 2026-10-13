#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/localscratch/USER/qwen36_openhands/hf_cache
cd /localscratch/USER || exit 1
LOG=/localscratch/USER/lcb_rebuild.log
echo "=== rebuilding official LiveCodeBench ===" > "$LOG"
rm -rf LiveCodeBench
git clone -q --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git >> "$LOG" 2>&1
cd LiveCodeBench || exit 1
uv venv --python 3.11 .venv >> "$LOG" 2>&1
uv pip install --python .venv/bin/python -e . >> "$LOG" 2>&1
# their loader is a HF dataset script -> needs datasets<3
uv pip install --python .venv/bin/python "datasets==2.21.0" "fsspec<=2024.6.1" >> "$LOG" 2>&1
.venv/bin/python -c "import datasets;print('datasets',datasets.__version__)" >> "$LOG" 2>&1
echo LCB_REBUILD_DONE >> "$LOG"
