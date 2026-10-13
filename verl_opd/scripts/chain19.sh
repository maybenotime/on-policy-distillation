#!/bin/bash
# cs-venus-19 chain: Qwen3-1.7B <- Code-Step300, then <- RLCR, on GPUs 0,1,2
# (2 actor + 1 teacher). GPU 3 stays free for merge/eval. Same recipe as chain_teachers.sh.
set -u
source /home/sya225/opd/env.sh none
COMMON=(
  STUDENT_MODEL=Qwen/Qwen3-1.7B
  DATA_DIR=$OPD_HOME/verl_data/ot3_code30k
  TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1
  ACTOR_LR=5e-6 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=8192
  DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64
  NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1
  SAVE_FREQ=100 HF_HUB_OFFLINE=1
)
run () {   # tag  teacher  gpus
  echo "######## START $1 $2 gpus=$3 $(date -Is) ########"
  CUDA_VISIBLE_DEVICES=$3 RAY_TMPDIR=/tmp/ray_$1 \
  env "${COMMON[@]}" TEACHER_MODEL="$2" TEACHER_GPU_MEM_UTIL=0.45 \
      CKPT_DIR=$SCRATCH/checkpoints/opd_code_17b_from$1 \
      bash $OPD_HOME/scripts/run_opd_code.sh \
      >> $SCRATCH/logs/train_17b_from$1.log 2>&1
  echo "######## END $1 rc=$? $(date -Is) ########"
}
run K4Bcode300  Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300  0,1,2
run RLCR4B      resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step 0,1,2
echo "CHAIN_19_DONE $(date -Is)"
