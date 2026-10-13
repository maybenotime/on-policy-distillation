#!/bin/bash
# job 265672 (cs-venus-19): OPD  student DeepSeek-R1-Distill-Qwen-1.5B <- teacher Skywork/Skywork-OR1-Math-7B (GPUs 2,3 student; GPU 1 teacher)
# data DAPO-Math-17k (17,398 unique prompts, 1 epoch = 271 steps), k1 + PG with the k1_probe plugin
# (per-step overlap16 / rho / ForkOverlap / dH), lr 1e-6, 16k responses, ckpt every 50 steps.
# GPUs 0,1 student, 2 teacher.
source /home/sya225/opd/env.sh none; S=/home/sya225/opd/scripts; export VERL_TEACHER_KEEP_SAMPLED=1
TAG=ds15b_fromSky7B_dapo_16k
echo "######## START $TAG $(date -Is) ########"
CUDA_VISIBLE_DEVICES=2,3,1 RAY_TMPDIR=/tmp/ray_sky \
env STUDENT_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B TEACHER_MODEL=Skywork/Skywork-OR1-Math-7B \
    DATA_DIR=/home/sya225/opd/verl_data/dapo_math17k \
    TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 ACTOR_LR=1e-6 MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    DISTILLATION_LOSS_MODE=k1_probe USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64 SAVE_FREQ=50 HF_HUB_OFFLINE=1 \
    NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=$SCRATCH/checkpoints/opd_$TAG \
    bash $S/run_opd_code.sh actor_rollout_ref.actor.calculate_entropy=True >> $SCRATCH/logs/train_$TAG.log 2>&1
echo "######## END $TAG rc=$? $(date -Is) ########"
