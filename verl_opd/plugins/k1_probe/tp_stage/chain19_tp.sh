#!/bin/bash
# bash chain19_tp.sh <TAG> <MODE shape|student|uniform|none> <A_MODEL (preferences; also the vLLM probe teacher)> <B_MODEL (shape source) or -> <GPUS> <RAY_TMPDIR> [GLOBAL_T=1.0] [STEPS=200]
# OPD with the synthetic teacher A ⊕ shape(B): student Qwen/Qwen3-1.7B (non-thinking), DAPO-Math-17k, k1+PG (k1_transplant),
# lr 1e-6, 16k, batch 64, ckpt every 50 -- identical to chain05_q3.sh except the loss mode.  SMOKE=1 -> 2 steps, batch 8, 512 tok.
source /home/sya225/opd/env.sh none; S=/home/sya225/opd/scripts; export VERL_TEACHER_KEEP_SAMPLED=1 RAY_worker_register_timeout_seconds=600
TAG=$1; MODE=$2; A=$3; B=$4; GPUS=$5; RAYD=$6; GT=${7:-1.0}; STEPS=${8:-200}
export TRANSPLANT_MODE=$MODE TRANSPLANT_LOGITS_MODEL="$A" TRANSPLANT_SHAPE_MODEL="$B" TRANSPLANT_GLOBAL_T=$GT TRANSPLANT_CHUNK=${TRANSPLANT_CHUNK:-1024}
STEP_ARG="trainer.total_training_steps=$STEPS"; [ "${SMOKE:-0}" = "1" ] && STEP_ARG=""
echo "######## START $TAG mode=$MODE A=$A B=$B T=$GT gpus=$GPUS smoke=${SMOKE:-0} $(date -Is) ########"
CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR=$RAYD \
env STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL="$A" DATA_DIR=/home/sya225/opd/verl_data/dapo_math17k \
    TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 ACTOR_LR=1e-6 MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    DISTILLATION_LOSS_MODE=k1_transplant USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64 SAVE_FREQ=${SAVE_FREQ:-50} HF_HUB_OFFLINE=1 \
    NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=$SCRATCH/checkpoints/opd_$TAG \
    bash $S/run_opd_code.sh actor_rollout_ref.actor.calculate_entropy=True $STEP_ARG ray_kwargs.ray_init.num_cpus=32 >> $SCRATCH/logs/train_$TAG.log 2>&1
echo "######## END $TAG rc=$? $(date -Is) ########"
