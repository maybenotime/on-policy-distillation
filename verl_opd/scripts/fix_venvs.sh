#!/bin/bash
# Repair venvs orphaned by the /home -> BeeGFS migration.
#
# All three venvs were created by uv, whose interpreter lived in
# ~/.local/share/uv/python/... on the OLD per-node /home. That home is now
# shadowed by the shared BeeGFS mount, so bin/python is a dangling symlink and
# every venv fails with "No such file or directory" (exit 127). uv itself is
# gone too (~/.local/bin/uv).
#
# site-packages survived on /localscratch, and the system interpreter is
# CPython 3.12.3 vs the venvs' 3.12.x -- same minor version, so the cp312
# compiled extensions load unchanged. Repoint the venvs at /usr/bin/python3.12.
set -x
SYS=/usr/bin/python3.12
for V in /home/USER/opd/envs/verl_opd /home/USER/opd/envs/code_eval_env /home/USER/opd/envs/evalplus_env; do
    [ -d "$V" ] || { echo "skip missing $V"; continue; }
    echo "=== repairing $V ==="
    for name in python python3 python3.12; do
        ln -sfn "$SYS" "$V/bin/$name"
    done
    # pyvenv.cfg must agree, or site.py resolves the old prefix
    python3 - "$V" <<'PY'
import sys, pathlib, re
v = pathlib.Path(sys.argv[1])
cfg = v / "pyvenv.cfg"
lines = []
for ln in cfg.read_text().splitlines():
    if ln.startswith("home ="):
        ln = "home = /usr/bin"
    elif ln.startswith("base-executable"):
        ln = "base-executable = /usr/bin/python3.12"
    lines.append(ln)
cfg.write_text("\n".join(lines) + "\n")
print("patched", cfg)
PY
    "$V/bin/python" -c "import sys; print('  python', sys.version.split()[0], 'prefix', sys.prefix)" 2>&1 | tail -2
done
echo "=== import smoke test ==="
/home/USER/opd/envs/code_eval_env/bin/python -c "
import vllm, transformers, datasets, evalplus
print('code_eval_env OK: vllm', vllm.__version__, '| transformers', transformers.__version__)
" 2>&1 | tail -3
/home/USER/opd/envs/verl_opd/bin/python -c "
import torch, verl, vllm, flash_attn
print('verl_opd OK: torch', torch.__version__, '| verl', verl.__version__, '| flash_attn', flash_attn.__version__)
print('cuda available:', torch.cuda.is_available())
" 2>&1 | tail -3
echo "FIX_VENVS_DONE"
