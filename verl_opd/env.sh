# Portable OPD environment (node-*). Source this first on any node.
# Code / venvs / data live under $OPD_HOME (shared /home); only node-local
# state -- model cache, checkpoints, logs -- goes to $SCRATCH.
#   source /home/USER/opd/env.sh          # + activates the training venv
#   source /home/USER/opd/env.sh none     # env vars only
export OPD_HOME=/home/USER/opd
export SCRATCH=/localscratch/USER
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
