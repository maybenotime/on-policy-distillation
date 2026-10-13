#!/bin/bash
# Sequential OPD runs of Qwen3-1.7B against RL-tuned Qwen3-4B teachers, 3 GPUs
# per chain (2 actor + 1 teacher), same recipe as the 4B/8B/14B runs.
#   bash chain_teachers.sh A   -> GPUs 3,4,5 : K4Bmath1200  K4Bcode300  RLCR4B
#   bash chain_teachers.sh B   -> GPUs 0,1,2 : waits for the 4B resume, then K4Bcode1200  K4Bmath500
# Each run has its own CKPT_DIR, so verl's resume_mode=auto restarts a killed run
# from its last global_step_* if the chain is relaunched.
set -u
CHAIN=$1
COMMON=(
  STUDENT_MODEL=Qwen/Qwen3-1.7B
  DATA_DIR=/home/sya225/opd/verl_data/ot3_code30k
  TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1
  ACTOR_LR=5e-6 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=8192
  DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64
  NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1
  SAVE_FREQ=100
)
run () {   # tag  teacher  gpus
  echo "######## START $1 $2 gpus=$3 $(date -Is) ########"
  CUDA_VISIBLE_DEVICES=$3 RAY_TMPDIR=/tmp/ray_$1 \
  env "${COMMON[@]}" TEACHER_MODEL="$2" TEACHER_GPU_MEM_UTIL=0.45 \
      CKPT_DIR=/localscratch/sya225/checkpoints/opd_code_17b_from$1 \
      bash /home/sya225/opd/scripts/run_opd_code.sh \
      >> /localscratch/sya225/logs/train_17b_from$1.log 2>&1
  echo "######## END $1 rc=$? $(date -Is) ########"
}
case $CHAIN in
  A)
    run K4Bmath1200 Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200 3,4,5
    run K4Bcode300  Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300  3,4,5
    run RLCR4B      resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step 3,4,5
    ;;
  B)
    echo "waiting for 4B resume to finish $(date -Is)"
    until grep -q "END 4b(resume)" /localscratch/sya225/logs/resume_4b_wrapper.log 2>/dev/null; do sleep 120; done
    sleep 60   # let Ray/vLLM release GPUs 0-2
    run K4Bcode1200 Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 0,1,2
    run K4Bmath500  Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500  0,1,2
    ;;
esac
echo "CHAIN_${CHAIN}_DONE $(date -Is)"
