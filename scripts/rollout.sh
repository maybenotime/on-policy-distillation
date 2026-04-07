STUDENT_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"

export HF_HOME=/home/sya225/.cache/huggingface
export MODELSCOPE_CACHE=/home/sya225/.cache/modelscope
export TORCH_HOME=/home/sya225/.cache/torch
export VLLM_CACHE_ROOT=/home/sya225/.cache/vllm
export TMPDIR=/home/sya225/.cache/tmp
export HF_DATASETS_CACHE=/home/sya225/.cache/huggingface/datasets

mkdir -p $HF_HOME $MODELSCOPE_CACHE $TORCH_HOME $VLLM_CACHE_ROOT $TMPDIR

swift rollout \
    --model "$STUDENT_MODEL" \
    --vllm_data_parallel_size 1 \
    --use_hf true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --padding_free true \
    --host 0.0.0.0 \
    --port 8000 \
    --load_from_cache_file false
