#!/bin/bash
# Plan A: two OPD runs back to back, 6 GPUs each (4 actor + 2 teacher).
#
# Everything is copied verbatim from the Qwen3-1.7B <- Qwen3-8B run so the only
# variable is TEACHER_MODEL. NGPUS_PER_NODE goes 2 -> 4, which changes only the
# data-parallel sharding: train_batch_size == ppo_mini_batch_size == 64 still
# means exactly one update per step over the same 64 prompts, so the comparison
# holds. total_epochs=1 lands on 424 steps on its own (30k minus the 9.5% that
# filter_overlong_prompts drops = 27,136 / 64).
set -u
COMMON=(
  STUDENT_MODEL=Qwen/Qwen3-1.7B
  DATA_DIR=/home/sya225/opd/verl_data/ot3_code30k
  TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1
  ACTOR_LR=5e-6 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=8192
  DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64
  NGPUS_PER_NODE=4 TEACHER_WORLD_SIZE=2 TEACHER_TP=2
  SAVE_FREQ=100
)

run () {   # tag  teacher
  echo "######## START $1  teacher=$2  $(date -Is) ########"
  env "${COMMON[@]}" TEACHER_MODEL="$2" \
      CKPT_DIR=/localscratch/sya225/checkpoints/opd_code_17b_from$1 \
      bash /home/sya225/opd/scripts/run_opd_code.sh \
      > /localscratch/sya225/logs/train_17b_from$1.log 2>&1
  echo "######## END   $1  rc=$?  $(date -Is) ########"
  df -h /localscratch | tail -1
}

run 4b  Qwen/Qwen3-4B
run 14b Qwen/Qwen3-14B
echo "TWO_TEACHERS_DONE $(date -Is)"
