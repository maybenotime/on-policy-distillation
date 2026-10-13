#!/bin/bash
# Resume the Qwen3-1.7B <- Qwen3-4B OPD run from global_step_300 (died at 391:
# host OOM from concurrent LCB scoring killed vLLM EngineCore).
# Must keep NGPUS_PER_NODE=2: the FSDP checkpoint is sharded world_size=2.
set -u
COMMON=(
  STUDENT_MODEL=Qwen/Qwen3-1.7B
  DATA_DIR=/home/USER/opd/verl_data/ot3_code30k
  TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1
  ACTOR_LR=5e-6 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=8192
  DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64
  NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1
  SAVE_FREQ=100
)
echo "######## RESUME 4b from step $(cat /localscratch/USER/checkpoints/opd_code_17b_from4b/latest_checkpointed_iteration.txt) $(date -Is) ########"
CUDA_VISIBLE_DEVICES=0,1,2 RAY_TMPDIR=/tmp/ray_4b \
env "${COMMON[@]}" TEACHER_MODEL=Qwen/Qwen3-4B TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=/localscratch/USER/checkpoints/opd_code_17b_from4b \
    bash /home/USER/opd/scripts/run_opd_code.sh \
    >> /localscratch/USER/logs/train_17b_from4b.log 2>&1
echo "######## END 4b(resume) rc=$? $(date -Is) ########"
