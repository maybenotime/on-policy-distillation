STUDENT_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"

SLURM_TMPDIR="${SLURM_TMPDIR:-/localscratch/sya225}"

export HF_HOME=$SLURM_TMPDIR/.cache/huggingface
export MODELSCOPE_CACHE=$SLURM_TMPDIR/.cache/modelscope
export TORCH_HOME=$SLURM_TMPDIR/.cache/torch  
export VLLM_CACHE_ROOT=$SLURM_TMPDIR/.cache/vllm
export TMPDIR=$SLURM_TMPDIR/tmp

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
