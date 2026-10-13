#!/bin/bash
# merge run A s50 (CPU) and sample 24 DAPO prompts on GPU 2 (shares the GPU with run B's teacher; 0.3 util)
source /home/sya225/opd/env.sh none; export PATH="$CODE_EVAL_ENV/bin:$PATH"
CK=$SCRATCH/checkpoints/opd_q3i17_fromQ8Bdapo_dapo_16k/global_step_50/actor; HF=$SCRATCH/checkpoints/hf_Q8Bdapo_step50
if [ ! -s $HF/config.json ]; then
  CUDA_VISIBLE_DEVICES=2 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('Qwen/Qwen3-1.7B').save_pretrained('$HF')"
fi
echo "MERGED $(date -Is)"
CUDA_VISIBLE_DEVICES=2 "$CODE_EVAL_ENV/bin/python" /home/sya225/opd/scripts/sample_ckpt.py $HF /home/sya225/opd/verl_data/dapo_math17k/train.parquet 24 16384 0.3 $SCRATCH/logs/sample_qA50.json 2>&1 | grep -v "INFO\|WARNING\|it/s\|Processed prompts" | tail -20
echo "SAMPLE_DONE $(date -Is)"
