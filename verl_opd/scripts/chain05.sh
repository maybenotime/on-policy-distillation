#!/bin/bash
# 05: wait for 4B to finish -> 4B merge+eval on GPU 0 ; Code-1200 4 student + 1 teacher on GPUs 1-5
source /home/USER/opd/env.sh none
L=/localscratch/USER/logs
until grep -q "END 4b(resume)" $L/resume_4b_wrapper.log 2>/dev/null; do sleep 30; done
echo "4B finished $(date -Is)"; sleep 60

COMMON=(
  STUDENT_MODEL=Qwen/Qwen3-1.7B
  DATA_DIR=/home/USER/opd/verl_data/ot3_code30k
  TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1
  ACTOR_LR=5e-6 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=8192
  DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64
  SAVE_FREQ=100 HF_HUB_OFFLINE=1
)
run () {   # tag teacher gpus ngpus teacher_ws [extra env...]
  local tag=$1 teacher=$2 gpus=$3 ng=$4 tws=$5; shift 5
  echo "######## START $tag $teacher gpus=$gpus student=$ng teacher=$tws $(date -Is) ########"
  CUDA_VISIBLE_DEVICES=$gpus RAY_TMPDIR=/tmp/ray_$tag \
  env "${COMMON[@]}" NGPUS_PER_NODE=$ng TEACHER_WORLD_SIZE=$tws TEACHER_TP=1 \
      TEACHER_MODEL="$teacher" TEACHER_GPU_MEM_UTIL=0.45 \
      CKPT_DIR=$SCRATCH/checkpoints/opd_code_17b_from$tag "$@" \
      bash /home/USER/opd/scripts/run_opd_code.sh \
      >> $SCRATCH/logs/train_17b_from$tag.log 2>&1
  local rc=$?
  echo "######## END $tag rc=$rc $(date -Is) ########"
  return $rc
}
T=Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200
rm -rf $SCRATCH/checkpoints/opd_code_17b_fromK4Bcode1200
mv $SCRATCH/logs/train_17b_fromK4Bcode1200.log $SCRATCH/logs/train_17b_fromK4Bcode1200.killed.log 2>/dev/null

# 4B eval on GPU 0, in background
( bash /home/USER/opd/scripts/post_run.sh opd-1.7b-from4b $SCRATCH/checkpoints/opd_code_17b_from4b 0 8030 \
    > $SCRATCH/logs/post_run_4b.log 2>&1; echo "4B_EVAL_DONE rc=$? $(date -Is)" ) &

run K4Bcode1200 $T 1,2,3,4,5 4 1
wait
echo CHAIN_05_DONE
