#!/bin/bash -l
#SBATCH -J acc-eval
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=7:59:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Generate n_rollouts × n_prompts completions with a single vLLM model,
# math_verify-check each, save pass@1/2/4 to results/pass_at_k/<tag>.json.
#
# Required env:
#   MODEL_ID         — HF id (e.g. Qwen/Qwen2.5-3B-Instruct)
#   OUTPUT_TAG       — short tag for the output file
# Optional:
#   N_PROMPTS=10000  — first N rows from PROMPTS_JSONL
#   N_ROLLOUTS=4
#   TEMPERATURE=0.7

set -e
: "${MODEL_ID:?MODEL_ID required}"
: "${OUTPUT_TAG:?OUTPUT_TAG required}"
N_PROMPTS="${N_PROMPTS:-10000}"
N_ROLLOUTS="${N_ROLLOUTS:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
ENABLE_THINKING="${ENABLE_THINKING:-none}"
PROMPTS_JSONL="${PROMPTS_JSONL:-/project/aip-angelx/sya225/on-policy-distillation/data/prompts_math_with_answers.jsonl}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
START_IDX="${START_IDX:-0}"

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

mkdir -p logs results/pass_at_k

export HF_HOME=/scratch/sya225/hf_cache
export TRANSFORMERS_CACHE=/scratch/sya225/hf_cache
export MODELSCOPE_CACHE=/scratch/sya225/hf_cache/modelscope
export TORCH_HOME=/scratch/sya225/hf_cache/torch
export VLLM_CACHE_ROOT=/scratch/sya225/hf_cache/vllm
export TMPDIR=/scratch/sya225/hf_cache/tmp
export HF_DATASETS_CACHE=/tmp/sya225_ds_cache_$SLURM_JOB_ID
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$HF_DATASETS_CACHE"

echo "===================================================="
echo "[pass-at-k] MODEL_ID=$MODEL_ID  OUTPUT_TAG=$OUTPUT_TAG"
echo "[pass-at-k] N_PROMPTS=$N_PROMPTS N_ROLLOUTS=$N_ROLLOUTS T=$TEMPERATURE"
echo "===================================================="

source eval_test/bin/activate
python source/compute_pass_at_k.py \
    --model "$MODEL_ID" \
    --prompts_jsonl "$PROMPTS_JSONL" \
    --n_prompts "$N_PROMPTS" \
    --n_rollouts "$N_ROLLOUTS" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --enable_thinking "$ENABLE_THINKING" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --data_parallel_size "$DATA_PARALLEL_SIZE" \
    --start_idx "$START_IDX" \
    --output "results/correctness/${OUTPUT_TAG}.json"

echo "[pass-at-k] DONE"
