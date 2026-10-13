#!/bin/bash
# Wipe every evaluation artefact and restart from zero.
# Trained models and training checkpoints are deliberately NOT touched.
set -x
# 1. stop everything (kill by PID; `pkill -f` would match this script's own
#    srun command line and take the step down with it)
for pat in eval_v2.sh eval_one_model.sh gen2.py api_server "scripts/run.py"; do
    for pid in $(pgrep -u USER -f "$pat"); do kill -9 "$pid" 2>/dev/null; done
done
sleep 15
pgrep -u USER -c -f "gen2.py|api_server|eval_v2" || echo "all stopped"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' '|'; echo

# 2. wipe eval artefacts only
rm -rf /home/USER/opd/bench/official \
       /home/USER/opd/bench/sampled \
       /home/USER/opd/matharena/outputs
rm -f /localscratch/USER/logs/v2_*.log /localscratch/USER/logs/evalall_*.log \
      /localscratch/USER/logs/vllm_*.log
mkdir -p /home/USER/opd/bench/official
echo "--- kept ---"
ls -d /localscratch/USER/checkpoints/hf_opd*_step424 /localscratch/USER/checkpoints/opd_code_* 2>/dev/null

# 3. lift the concurrency throttle: measured Running=64/Waiting=0 with GPU KV
#    cache at only 26-30%, i.e. the harness was starving the GPU, not the reverse.
python3 - <<'PY'
import pathlib
d = pathlib.Path("/home/USER/opd/matharena/configs/models/local")
for p in sorted(d.glob("*.yaml")):
    s = p.read_text()
    if "concurrent_requests: 64" in s:
        p.write_text(s.replace("concurrent_requests: 64", "concurrent_requests: 256"))
        print("concurrency 64->256:", p.name)
PY
echo "RESET_DONE $(date -Is)"
