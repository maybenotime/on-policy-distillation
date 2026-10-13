#!/bin/bash
# Stage 1: copy node-local (cs-venus-05) environments + code into a portable
# /home tree. Models/checkpoints/logs stay node-local under /localscratch.
set -u
S=/localscratch/sya225
H=/home/sya225/opd
mkdir -p $H/envs $H/scripts
t0=$(date +%s)
log () { echo "[$(( $(date +%s) - t0 ))s] $*"; }

log "verl_opd venv"
rsync -a --info=progress2 $S/verl_opd/ $H/envs/verl_opd/ 2>&1 | tail -1
log "code + data"
rsync -a --exclude .venv --exclude .git --exclude logs $S/LiveCodeBench/ $H/LiveCodeBench/
rsync -a --exclude .venv --exclude .git --exclude logs --exclude paper_search_cache $S/matharena/ $H/matharena/
rsync -a --exclude .git $S/verl_repo/ $H/verl_repo/
rsync -a $S/verl_data/ $H/verl_data/
rsync -a $S/bench/ $H/bench/
rsync -a $S/fork/ $H/fork/
rsync -a $S/results/ $H/results/
cp -a $S/*.sh $S/*.py $S/*.json $S/*.txt $H/scripts/ 2>/dev/null
cp -a $S/gen2.py.bak $H/scripts/ 2>/dev/null
log "sizes"; du -sh $H/* | sort -h
df -h /home | tail -1
log "DONE"
