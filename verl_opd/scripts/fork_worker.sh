#!/bin/bash
# One student: rollout -> student entropy -> teacher entropy, all on one GPU.
# Separate processes so vLLM's KV cache is released before the HF forward passes.
TAG=$1; MODEL=$2; GPU=$3
E=/home/sya225/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_LOGGING_LEVEL=WARNING
D=/home/sya225/opd/fork
TEACHER=Qwen/Qwen3-8B
mkdir -p $D

echo "######## [$TAG] rollout $(date -Is) ########"
[ -s "$D/roll_$TAG.jsonl" ] || "$E/bin/python" /home/sya225/opd/scripts/fork_rollout.py \
    --model "$MODEL" --n 1000 --seed 0 --max_tokens 4096 --out "$D/roll_$TAG.jsonl"

echo "######## [$TAG] student entropy $(date -Is) ########"
[ -s "$D/ent_student_$TAG.npz" ] || "$E/bin/python" /home/sya225/opd/scripts/fork_entropy.py \
    --model "$MODEL" --rollouts "$D/roll_$TAG.jsonl" --out "$D/ent_student_$TAG.npz"

echo "######## [$TAG] teacher entropy $(date -Is) ########"
[ -s "$D/ent_teacher_$TAG.npz" ] || "$E/bin/python" /home/sya225/opd/scripts/fork_entropy.py \
    --model "$TEACHER" --rollouts "$D/roll_$TAG.jsonl" --out "$D/ent_teacher_$TAG.npz"

echo "######## [$TAG] DONE $(date -Is) ########"
