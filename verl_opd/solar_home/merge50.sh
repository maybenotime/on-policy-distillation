set -u
source /home/sya225/opd/env.sh none
CK=$SCRATCH/checkpoints/opd_ds15b_fromJustRL_dapo_16k/global_step_50/actor
HF=$SCRATCH/checkpoints/hf_ds15b_fromJustRL_step50
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
if [ ! -s $HF/config.json ]; then
  echo "MERGE_START $(date -Is)"
  CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
      --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -2
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
ls -la $HF | head -8
echo "MERGE_DONE $(date -Is)"
