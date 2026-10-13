#!/bin/bash
# job 269604 (cs-venus-05): OPD  student DeepSeek-R1-Distill-Qwen-1.5B <- teacher agentica-org/DeepScaleR-1.5B-Preview
# DAPO-Math-17k, k1 + PG (k1_probe plugin), lr 1e-6, 16k responses, ckpt every 50, stops by itself at step 200.
# GPUs 0,1,2 (first two = student, third = teacher).
source /home/sya225/opd/env.sh none; S=/home/sya225/opd/scripts; export VERL_TEACHER_KEEP_SAMPLED=1 RAY_worker_register_timeout_seconds=600
TAG=ds15b_fromDeepScaleR_dapo_16k; TEACHER="agentica-org/DeepScaleR-1.5B-Preview"
echo "######## START $TAG $(date -Is) ########"
CUDA_VISIBLE_DEVICES=0,1,2 RAY_TMPDIR=/tmp/ray_dsc \
env STUDENT_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B TEACHER_MODEL="$TEACHER" \
    DATA_DIR=/home/sya225/opd/verl_data/dapo_math17k \
    TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 ACTOR_LR=1e-6 MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    DISTILLATION_LOSS_MODE=k1_probe USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64 SAVE_FREQ=50 HF_HUB_OFFLINE=1 \
    NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=$SCRATCH/checkpoints/opd_$TAG \
    bash $S/run_opd_code.sh actor_rollout_ref.actor.calculate_entropy=True trainer.total_training_steps=200 ray_kwargs.ray_init.num_cpus=32 >> $SCRATCH/logs/train_$TAG.log 2>&1
echo "######## END $TAG rc=$? $(date -Is) ########"
