#!/bin/bash
# Stage 2: copy the two eval venvs, relocate all venvs, rewrite hardcoded paths
# in the scripts so the tree under /home/sya225/opd is node-independent.
set -u
S=/localscratch/sya225
H=/home/sya225/opd
t0=$(date +%s); log () { echo "[$(( $(date +%s) - t0 ))s] $*"; }

log "code_eval_env venv"
rsync -a $S/code_eval_env/ $H/envs/code_eval_env/
log "matharena venv"
rsync -a $S/matharena/.venv/ $H/envs/matharena_venv/
ln -sfn ../envs/matharena_venv $H/matharena/.venv

relocate () {   # old_prefix new_prefix venv_dir
  local old=$1 new=$2 v=$3
  grep -lZ "$old" "$v"/bin/* 2>/dev/null | xargs -0 -r sed -i "s#$old#$new#g"
  echo "  relocated $v: $(grep -l "$new" "$v"/bin/* | wc -l) files"
}
log "relocate venvs"
relocate $S/verl_opd           $H/envs/verl_opd        $H/envs/verl_opd
relocate $S/code_eval_env      $H/envs/code_eval_env   $H/envs/code_eval_env
relocate $S/matharena/.venv    $H/envs/matharena_venv  $H/envs/matharena_venv

log "rewrite script paths"
cd $H/scripts
# things that now live in /home
for d in verl_opd code_eval_env evalplus_env; do sed -i "s#$S/$d#$H/envs/$d#g" *.sh *.py; done
for d in LiveCodeBench matharena verl_repo verl_data bench fork results; do sed -i "s#$S/$d\b#$H/$d#g" *.sh *.py; done
# the LCB dataset cache used to live inside an unrelated venv
sed -i "s#$S/qwen36_openhands/hf_cache/datasets#$S/hf_cache/datasets#g" *.sh *.py
# scripts calling sibling scripts by absolute path
for f in *.sh *.py; do sed -i "s#$S/$f\b#$H/scripts/$f#g" *.sh *.py; done
sed -i "s#$S/env_verl_opd.sh#$H/env.sh#g" *.sh *.py
# state that stays node-local: hf_cache checkpoints logs merged -> untouched
echo "remaining /localscratch refs:"; grep -ho "$S/[A-Za-z0-9_.-]*" *.sh *.py | sort | uniq -c | sort -rn

log "env.sh"
cat > $H/env.sh <<'EOS'
# Portable OPD environment (cs-venus-*). Source this first on any node.
# Everything that is code/venv/data lives under $OPD_HOME (shared /home);
# only node-local state -- model cache, checkpoints, logs -- goes to $SCRATCH.
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
# run_opd_code.sh / run_opd_math.sh sourced the old env file; point them at env.sh
sed -i "s#source $S/env_verl_opd.sh#source $H/env.sh#" $H/scripts/*.sh
log "sizes"; du -sh $H/envs/* $H | sort -h; df -h /home | tail -1
log "DONE"
