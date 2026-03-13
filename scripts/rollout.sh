mkdir -p $SLURM_TMPDIR/sya225/.cache/model

export HUGGINGFACE_HUB_CACHE="$SLURM_TMPDIR/sya225/.cache/model"

swift rollout \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --vllm_data_parallel_size 1 \
    --use_hf true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --padding_free true \
    --host 0.0.0.0 \
    --port 8000 \
    --load_from_cache_file false
