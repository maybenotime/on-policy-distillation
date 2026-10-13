#!/bin/bash
V=/home/sya225/opd/envs/verl_opd
P=$("$V/bin/python" -c "import verl,pathlib;print(pathlib.Path(verl.__file__).parent)")
echo "verl root: $P"
echo "===== 顶层模块 ====="; ls "$P"
echo; echo "===== trainer ====="; ls "$P/trainer" 2>/dev/null
echo; echo "===== workers ====="; ls "$P/workers" 2>/dev/null
echo; echo "===== 搜 distill / kd / teacher ====="
grep -ril -E "distill" "$P" --include=*.py --include=*.yaml 2>/dev/null | head -20
echo "--- config 里的 teacher ---"
grep -rl -E "teacher" "$P" --include=*.yaml 2>/dev/null | head -10
