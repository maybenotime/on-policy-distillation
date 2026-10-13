#!/bin/bash
# Restore uv + its bundled CPython, then repoint the venvs back at it.
#
# Pointing the venvs at /usr/bin/python3.12 fixed the dangling-symlink breakage
# but broke Triton: its runtime JIT compiles cuda_utils.c and needs Python.h,
# which the system interpreter lacks (no python3.12-dev, and no root to add it).
# uv's CPython ships its own headers. This now lands on the shared BeeGFS /home,
# so unlike before it will be visible from every compute node.
#
# uv comes from the prebuilt tarball -- installing it from PyPI compiles Rust
# from source and takes 10+ minutes.
set -x
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/.local/bin"

if ! command -v uv >/dev/null; then
    curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
      | tar -xz -C /tmp
    mv /tmp/uv-x86_64-unknown-linux-gnu/uv /tmp/uv-x86_64-unknown-linux-gnu/uvx "$HOME/.local/bin/"
fi
uv --version

export UV_CACHE_DIR=/localscratch/sya225/uv_cache
uv python install 3.12 2>&1 | tail -3

PY=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*-linux-x86_64-gnu 2>/dev/null | head -1)
echo "installed interpreter: $PY"
ls "$PY/include/python3.12/Python.h" || { echo "NO HEADERS"; exit 1; }

for V in /home/sya225/opd/envs/verl_opd /home/sya225/opd/envs/code_eval_env /home/sya225/opd/envs/evalplus_env; do
    [ -d "$V" ] || continue
    for n in python python3 python3.12; do ln -sfn "$PY/bin/python3.12" "$V/bin/$n"; done
    python3 - "$V" "$PY" <<'PY'
import sys, pathlib
v, py = pathlib.Path(sys.argv[1]), sys.argv[2]
cfg = v / "pyvenv.cfg"
out = []
for ln in cfg.read_text().splitlines():
    if ln.startswith("home ="):
        ln = f"home = {py}/bin"
    elif ln.startswith("base-executable"):
        ln = f"base-executable = {py}/bin/python3.12"
    out.append(ln)
cfg.write_text("\n".join(out) + "\n")
PY
    "$V/bin/python" -c "import sys,sysconfig,os;h=sysconfig.get_paths()['include'];print(' ',os.path.basename('$V'),sys.version.split()[0],'headers:',os.path.exists(os.path.join(h,'Python.h')))"
done
echo "RESTORE_DONE"
