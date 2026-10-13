#!/bin/bash
# Stage 3 (replaces the tail of migrate2, whose grep over bin/* crawled on
# BeeGFS): relocate venvs by touching only shebang scripts + activate files,
# then rewrite script paths and write env.sh.
set -u
S=/localscratch/sya225
H=/home/sya225/opd
t0=$(date +%s); log () { echo "[$(( $(date +%s) - t0 ))s] $*"; }

relocate () {   # old_prefix new_prefix venv_dir
  local old=$1 new=$2 v=$3 n=0
  for f in "$v"/bin/*; do
    [ -L "$f" ] && continue
    case "$(basename "$f")" in activate*) ;; *) [ "$(head -c2 "$f" 2>/dev/null)" = "#!" ] || continue ;; esac
    if grep -q "$old" "$f"; then sed -i "s#$old#$new#g" "$f"; n=$((n+1)); fi
  done
  echo "  $v: rewrote $n files; stale refs left: $(grep -l "$old" "$v"/bin/activate "$v"/bin/ray "$v"/bin/vllm "$v"/bin/python* 2>/dev/null | wc -l)"
}
log "relocate venvs"
relocate $S/verl_opd           $H/envs/verl_opd        $H/envs/verl_opd
relocate $S/code_eval_env      $H/envs/code_eval_env   $H/envs/code_eval_env
relocate $S/matharena/.venv    $H/envs/matharena_venv  $H/envs/matharena_venv
ln -sfn ../envs/matharena_venv $H/matharena/.venv

log "rewrite script paths"
cd $H/scripts
for d in verl_opd code_eval_env evalplus_env; do sed -i "s#$S/$d#$H/envs/$d#g" *.sh *.py; done
for d in LiveCodeBench matharena verl_repo verl_data bench fork results; do sed -i "s#$S/$d\b#$H/$d#g" *.sh *.py; done
sed -i "s#$S/qwen36_openhands/hf_cache/datasets#$S/hf_cache/datasets#g" *.sh *.py
for f in *.sh *.py; do sed -i "s#$S/$f\b#$H/scripts/$f#g" *.sh *.py; done
sed -i "s#$S/env_verl_opd.sh#$H/env.sh#g" *.sh *.py
echo "remaining /localscratch refs:"; grep -ho "$S/[A-Za-z0-9_.-]*" *.sh *.py | sort | uniq -c | sort -rn

log "env.sh"
cat > $H/env.sh <<'EOS'
# Portable OPD environment (cs-venus-*). Source this first on any node.
# Code / venvs / data live under $OPD_HOME (shared /home); only node-local
# state -- model cache, checkpoints, logs -- goes to $SCRATCH.
#   source /home/sya225/opd/env.sh          # + activates the training venv
#   source /home/sya225/opd/env.sh none     # env vars only
export OPD_HOME=/home/sya225/opd
export SCRATCH=/localscratch/sya225
export VERL_VENV=$OPD_HOME/envs/verl_opd
export VERL_REPO=$OPD_HOME/verl_repo
export CODE_EVAL_ENV=$OPD_HOME/envs/code_eval_env
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export UV_CACHE_DIR=$SCRATCH/uv_cache
export PATH="$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
mkdir -p $SCRATCH/hf_cache $SCRATCH/checkpoints $SCRATCH/logs $SCRATCH/uv_cache
if [ "${1:-train}" = train ]; then source "$VERL_VENV/bin/activate"; fi
EOS
grep -n "source.*env" $H/scripts/run_opd_code.sh
log "DONE"; df -h /home | tail -1
