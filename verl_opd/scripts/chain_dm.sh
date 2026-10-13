#!/bin/bash
# bash chain_dm.sh <TAG> <STUDENT> <TEACHER> <DATA_DIR> <GPUS e.g. 0,1,2> <RAY_TMPDIR>   (first two GPUs student, third teacher)
# same recipe as the DAPO runs: k1 + PG (k1_probe), lr 1e-6, 16k responses, batch 64, ckpt every 50, stops at step 200.
source /home/sya225/opd/env.sh none; S=/home/sya225/opd/scripts; export VERL_TEACHER_KEEP_SAMPLED=1
TAG=$1; STU=$2; TEA=$3; DD=$4; GPUS=$5; RAYD=$6
echo "######## START $TAG data=$DD $(date -Is) ########"
CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR=$RAYD \
env STUDENT_MODEL=$STU TEACHER_MODEL=$TEA DATA_DIR=$DD \
    TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 ACTOR_LR=1e-6 MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    DISTILLATION_LOSS_MODE=k1_probe USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64 SAVE_FREQ=50 HF_HUB_OFFLINE=1 \
    NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=$SCRATCH/checkpoints/opd_$TAG \
    bash $S/run_opd_code.sh actor_rollout_ref.actor.calculate_entropy=True trainer.total_training_steps=200 >> $SCRATCH/logs/train_$TAG.log 2>&1
echo "######## END $TAG rc=$? $(date -Is) ########"
