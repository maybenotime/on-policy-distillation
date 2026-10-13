# Source this to get the verl OPD env on cs-venus-05.
#   source /home/sya225/opd/env.sh
export VERL_VENV=/home/sya225/opd/envs/verl_opd
export VERL_REPO=/home/sya225/opd/verl_repo
export HF_HOME=/localscratch/sya225/hf_cache
export UV_CACHE_DIR=/localscratch/sya225/uv_cache
export PATH="$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
# NOTE: no LD_PRELOAD needed. torch 2.13 bundles NCCL 2.29.7, measured
# identical (7.23 GB/s busbw, 3 GPUs) to the 2.30.7 build in
# /localscratch/sya225/nccl_new. These cards are PCIe-bound (no NVLink,
# all on NUMA node 1), so NCCL version is not the limiter here.
source "$VERL_VENV/bin/activate"
echo "verl OPD env | python $(python --version 2>&1 | cut -d' ' -f2) | verl $(python -c 'import verl;print(verl.__version__)' 2>/dev/null)"
