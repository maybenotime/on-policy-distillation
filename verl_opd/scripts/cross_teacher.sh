#!/bin/bash
# Same Qwen3-1.7B rollouts, three different teachers. Reusing one rollout file is
# what makes the teachers comparable: every entropy profile is measured at exactly
# the same token positions, so only the teacher differs.
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/USER/hf_cache
D=/home/USER/opd/fork
R=$D/roll_Qwen3-1.7B.jsonl

run_one () {   # tag  model  gpu
  echo "######## teacher=$1 $(date -Is) ########"
  CUDA_VISIBLE_DEVICES=$3 "$E/bin/python" /home/USER/opd/scripts/fork_entropy.py \
      --model "$2" --rollouts "$R" --out "$D/ent_T$1_Qwen3-1.7B.npz"
}

( run_one 4B Qwen/Qwen3-4B 0 ) > /localscratch/USER/logs/cross_4B.log 2>&1 &
P4=$!

(
  echo "######## downloading Qwen3-14B $(date -Is) ########"
  "$E/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-14B', allow_patterns=['*.json','*.safetensors','*.txt'])
print('downloaded to', p, flush=True)"
  run_one 14B Qwen/Qwen3-14B 1
) > /localscratch/USER/logs/cross_14B.log 2>&1 &
P14=$!

wait $P4 $P14
echo "CROSS_DONE $(date -Is)"
