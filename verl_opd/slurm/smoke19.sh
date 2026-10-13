#!/bin/bash
# Smoke test of the portable /home tree on a fresh node: 2 OPD steps.
source /home/sya225/opd/env.sh
export STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL=Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300
export DATA_DIR=/home/sya225/opd/verl_data/ot3_code30k
export TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 ACTOR_LR=5e-6
export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=8192
export DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64
export NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_GPU_MEM_UTIL=0.45 SAVE_FREQ=100
export CKPT_DIR=/localscratch/sya225/checkpoints/smoke19 RAY_TMPDIR=/tmp/ray_smoke
export CUDA_VISIBLE_DEVICES=0,1,2 HF_HUB_OFFLINE=1
rm -rf "$CKPT_DIR"
SMOKE=1 bash /home/sya225/opd/scripts/run_opd_code.sh
echo "######## SMOKE rc=$?"
