# vLLM rollout server for Qwen3 (non-thinking mode)
# Hardcoded --enable_thinking false so the served model emits answers without
# the <think>...</think> reasoning trace.

STUDENT_MODEL="${1:-Qwen/Qwen3-1.7B}"

CACHE_BASE="${CACHE_BASE:-/scratch/sya225/hf_cache}"
# Respect pre-set HF_HOME (caller may already point at an existing model cache).
export HF_HOME="${HF_HOME:-$CACHE_BASE}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_BASE/modelscope}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm}"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"

swift rollout \
    --model "$STUDENT_MODEL" \
    --enable_thinking false \
    --vllm_data_parallel_size 1 \
    --use_hf true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --padding_free true \
    --host 0.0.0.0 \
    --port 8000 \
    --load_from_cache_file false
