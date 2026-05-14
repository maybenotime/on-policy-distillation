STUDENT_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"

CACHE_BASE="${CACHE_BASE:-/home/sya225/.cache}"

export HF_HOME="${HF_HOME:-$CACHE_BASE/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_BASE/modelscope}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm}"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"

# When PREFIX_PREFILL=1, run a wrapper that monkey-patches Template.encode to
# accept raw prompt_token_ids via RolloutInferRequest.data_dict. Required for
# TP-BPOD (Phase 2 backward direction). Standard swift rollout otherwise.
ROLLOUT_ARGS=(
    --model "$STUDENT_MODEL"
    --vllm_data_parallel_size 1
    --use_hf true
    --torch_dtype bfloat16
    --attn_impl flash_attention_2
    --padding_free true
    --host 0.0.0.0
    --port 8000
    --load_from_cache_file false
)

if [ "${PREFIX_PREFILL:-0}" = "1" ]; then
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    export PYTHONPATH="${REPO_ROOT}/source:${PYTHONPATH:-}"
    exec python "${REPO_ROOT}/source/prefix_curriculum/rollout_with_prefill.py" "${ROLLOUT_ARGS[@]}"
else
    exec swift rollout "${ROLLOUT_ARGS[@]}"
fi
