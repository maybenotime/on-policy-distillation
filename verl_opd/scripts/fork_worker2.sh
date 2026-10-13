#!/bin/bash
# fork_worker.sh with the teacher as an argument: this student was distilled from
# Qwen3-14B, so its post-OPD alignment must be measured against that teacher.
TAG=$1; MODEL=$2; GPU=$3; TEACHER=$4
E=/home/sya225/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache
export CUDA_VISIBLE_DEVICES=$GPU VLLM_LOGGING_LEVEL=WARNING
D=/home/sya225/opd/fork
mkdir -p $D
echo "######## [$TAG] rollout $(date -Is) ########"
[ -s "$D/roll_$TAG.jsonl" ] || "$E/bin/python" /home/sya225/opd/scripts/fork_rollout.py \
    --model "$MODEL" --n 1000 --seed 0 --max_tokens 4096 --out "$D/roll_$TAG.jsonl"
# vLLM 0.27 can hang in EngineCore teardown after writing its output; the file is
# complete at that point, so a stuck rollout process is safe to kill by PID.
echo "######## [$TAG] student entropy $(date -Is) ########"
[ -s "$D/ent_student_$TAG.npz" ] || "$E/bin/python" /home/sya225/opd/scripts/fork_entropy.py \
    --model "$MODEL" --rollouts "$D/roll_$TAG.jsonl" --out "$D/ent_student_$TAG.npz"
echo "######## [$TAG] teacher entropy ($TEACHER) $(date -Is) ########"
[ -s "$D/ent_teacher_$TAG.npz" ] || "$E/bin/python" /home/sya225/opd/scripts/fork_entropy.py \
    --model "$TEACHER" --rollouts "$D/roll_$TAG.jsonl" --out "$D/ent_teacher_$TAG.npz"
echo "######## [$TAG] DONE $(date -Is) ########"
