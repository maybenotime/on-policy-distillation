#!/bin/bash
# bash chain19_ts.sh <TAG> <TEACHER> <GPUS> <RAY_TMPDIR> [STEPS=100]
# Plain k1_probe OPD (student Qwen/Qwen3-1.7B, DAPO-Math-17k, lr 1e-6, 16k, batch 64, ckpt every 50) with the
# token-selection knobs passed through the environment:  TOKEN_SELECT=none|mismatch|entropy|random|both_committed,
# TOKEN_FRAC (default 0.3).  SMOKE=1 -> 2 steps, batch 8, 512 tok.  Same recipe as chain05_q3.sh / chain19_q3i_m500.sh.
source /home/sya225/opd/env.sh none; S=/home/sya225/opd/scripts; export VERL_TEACHER_KEEP_SAMPLED=1 RAY_worker_register_timeout_seconds=600
TAG=$1; TEA=$2; GPUS=$3; RAYD=$4; STEPS=${5:-100}
export TOKEN_SELECT=${TOKEN_SELECT:-none} TOKEN_FRAC=${TOKEN_FRAC:-0.3} TOKEN_SEED=${TOKEN_SEED:-0}
STEP_ARG="trainer.total_training_steps=$STEPS"; [ "${SMOKE:-0}" = "1" ] && STEP_ARG=""
echo "######## START $TAG teacher=$TEA select=$TOKEN_SELECT frac=$TOKEN_FRAC gpus=$GPUS smoke=${SMOKE:-0} $(date -Is) ########"
CUDA_VISIBLE_DEVICES=$GPUS RAY_TMPDIR=$RAYD \
env STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL="$TEA" DATA_DIR=/home/sya225/opd/verl_data/dapo_math17k \
    TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 ACTOR_LR=1e-6 MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    DISTILLATION_LOSS_MODE=k1_probe USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64 SAVE_FREQ=${SAVE_FREQ:-50} HF_HUB_OFFLINE=1 \
    NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=$SCRATCH/checkpoints/opd_$TAG \
    bash $S/run_opd_code.sh actor_rollout_ref.actor.calculate_entropy=True $STEP_ARG ray_kwargs.ray_init.num_cpus=32 >> $SCRATCH/logs/train_$TAG.log 2>&1
echo "######## END $TAG rc=$? $(date -Is) ########"
