#!/bin/bash
# job 264964 slot B (GPUs 3-5): 1) resume code-data Math-1200 16k run (ckpt step 200), 2) math-data OPD 16k: 4B -> Math-500 -> RLCR
source /home/USER/opd/scripts/math_common.sh
echo "######## RESUME code-data K4Bmath1200_16k gpus=3,4,5 $(date -Is) ########"
CUDA_VISIBLE_DEVICES=3,4,5 RAY_TMPDIR=/tmp/ray_K4Bmath1200_16k_r \
env STUDENT_MODEL=Qwen/Qwen3-1.7B DATA_DIR=/home/USER/opd/verl_data/ot3_code30k \
    TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1 ACTOR_LR=5e-6 MAX_PROMPT_LENGTH=1024 \
    MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576 \
    DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64 SAVE_FREQ=100 HF_HUB_OFFLINE=1 \
    NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_MODEL=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200 TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=$SCRATCH/checkpoints/opd_code_17b_fromK4Bmath1200_16k \
    bash $S/run_opd_code.sh >> $SCRATCH/logs/train_17b_fromK4Bmath1200_16k.log 2>&1
echo "######## END resume code-data K4Bmath1200_16k rc=$? $(date -Is) ########"
for t in 4b K4Bmath500 RLCR4B; do train $t $(teacher_of $t) 3,4,5; done
echo "CHAIN_MATH_B_DONE $(date -Is)"
