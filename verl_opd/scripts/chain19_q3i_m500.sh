#!/bin/bash
# job 265672 (node-B): OPD  student Qwen/Qwen3-1.7B <- teacher Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500, both non-thinking (GPUs 1-3; GPU 0 runs the DS-7B s200 eval)
# same recipe as chain19_ds.sh: DAPO-Math-17k (271 steps), k1_probe (per-step overlap16/rho/FO/dH), lr 1e-6, 16k, ckpt every 50.
source /home/USER/opd/env.sh none; S=/home/USER/opd/scripts; export VERL_TEACHER_KEEP_SAMPLED=1
TAG=q3i17_fromK4Bmath500_dapo_16k
echo "######## START $TAG $(date -Is) ########"
CUDA_VISIBLE_DEVICES=2,3,1 RAY_TMPDIR=/tmp/ray_m500 \
env STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
    DATA_DIR=/home/USER/opd/verl_data/dapo_math17k \
    TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 ACTOR_LR=1e-6 MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    DISTILLATION_LOSS_MODE=k1_probe USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64 SAVE_FREQ=50 HF_HUB_OFFLINE=1 \
    NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=$SCRATCH/checkpoints/opd_$TAG \
    bash $S/run_opd_code.sh actor_rollout_ref.actor.calculate_entropy=True >> $SCRATCH/logs/train_$TAG.log 2>&1
echo "######## END $TAG rc=$? $(date -Is) ########"
