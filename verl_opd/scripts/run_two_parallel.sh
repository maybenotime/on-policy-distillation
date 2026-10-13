#!/bin/bash
# Plan B: the two teacher variants run CONCURRENTLY, 3 GPUs each
# (2 actor + 1 teacher), which is the same actor width as the Qwen3-8B baseline.
#
# Plan A (4 actor GPUs) was measured and abandoned: generation did NOT speed up
# (107.0 s both at 2 and at 4 actor GPUs) because step time is set by the longest
# sequence in each replica's batch, not by the token throughput -- splitting 64
# prompts into 4x16 still leaves a near-8192-token straggler in every replica.
# All it added was weight-sync to 4 rollout replicas: 143 s/step vs 122 s.
#
# Only TEACHER_MODEL differs from the Qwen3-1.7B <- Qwen3-8B run. The teacher is
# inference-only, so its TP/mem_util affect speed but not the objective.
set -u
COMMON=(
  STUDENT_MODEL=Qwen/Qwen3-1.7B
  DATA_DIR=/home/sya225/opd/verl_data/ot3_code30k
  TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1
  ACTOR_LR=5e-6 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=8192
  DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64
  NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1
  SAVE_FREQ=100
)

run () {   # tag  teacher  gpus  teacher_mem_util
  CUDA_VISIBLE_DEVICES=$3 RAY_TMPDIR=/tmp/ray_$1 \
  env "${COMMON[@]}" TEACHER_MODEL="$2" TEACHER_GPU_MEM_UTIL=$4 \
      CKPT_DIR=/localscratch/sya225/checkpoints/opd_code_17b_from$1 \
      bash /home/sya225/opd/scripts/run_opd_code.sh \
      > /localscratch/sya225/logs/train_17b_from$1.log 2>&1
  echo "######## END $1 rc=$? $(date -Is) ########"
}

echo "######## START both $(date -Is) ########"
# 14B needs a bigger slice: 28 GB of weights alone, vs 0.45*96 = 43 GB at TP=1.
run 4b  Qwen/Qwen3-4B  0,1,2 0.45 &
run 14b Qwen/Qwen3-14B 3,4,5 0.60 &
wait
echo "TWO_TEACHERS_DONE $(date -Is)"
