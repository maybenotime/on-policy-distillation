#!/bin/bash
# bash chain05_q3.sh <TAG> <TEACHER> <GPUS e.g. 3,4,5> <RAY_TMPDIR>   (job 269604, node-A)
# OPD: student Qwen/Qwen3-1.7B (non-thinking) <- TEACHER on DAPO-Math-17k, k1+PG (k1_probe), lr 1e-6, 16k, batch 64,
# ckpt every 50, stops at step 200.  Same recipe as chain19_q3i_m500.sh, plus the Ray fixes for node-A.
source /home/USER/opd/env.sh none; S=/home/USER/opd/scripts; export VERL_TEACHER_KEEP_SAMPLED=1 RAY_worker_register_timeout_seconds=600
TAG=$1; TEA=$2; GPUS=$3; RAYD=$4
echo "######## START $TAG teacher=$TEA gpus=$GPUS $(date -Is) ########"
CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR=$RAYD \
env STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL="$TEA" DATA_DIR=/home/USER/opd/verl_data/dapo_math17k \
    TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 ACTOR_LR=1e-6 MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    DISTILLATION_LOSS_MODE=k1_probe USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64 SAVE_FREQ=50 HF_HUB_OFFLINE=1 \
    NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=$SCRATCH/checkpoints/opd_$TAG \
    bash $S/run_opd_code.sh actor_rollout_ref.actor.calculate_entropy=True trainer.total_training_steps=200 ray_kwargs.ray_init.num_cpus=32 >> $SCRATCH/logs/train_$TAG.log 2>&1
echo "######## END $TAG rc=$? $(date -Is) ########"
