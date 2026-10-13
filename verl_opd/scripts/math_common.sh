# shared by chain05_mathA.sh / chain05_mathB.sh : OPD on ot3_math30k, 16k responses, 3 GPUs per run (student 2 + teacher 1)
source /home/USER/opd/env.sh none
S=/home/USER/opd/scripts
COMMON=(
  STUDENT_MODEL=Qwen/Qwen3-1.7B
  DATA_DIR=/home/USER/opd/verl_data/ot3_math30k
  TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=1
  ACTOR_LR=5e-6 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576
  DISTILLATION_LOSS_MODE=k1 USE_POLICY_GRADIENT=True DISTILLATION_TOPK=64
  SAVE_FREQ=50 HF_HUB_OFFLINE=1
)
train () {   # tag teacher gpus(e.g. 0,1,2 : student on first two, teacher on third)
  local tag=$1 teacher=$2 gpus=$3
  echo "######## START math16k $tag $teacher gpus=$gpus $(date -Is) ########"
  CUDA_VISIBLE_DEVICES=$gpus RAY_TMPDIR=/tmp/ray_math16k_$tag \
  env "${COMMON[@]}" NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 \
      TEACHER_MODEL="$teacher" TEACHER_GPU_MEM_UTIL=0.45 \
      CKPT_DIR=$SCRATCH/checkpoints/opd_math_17b_from${tag}_16k \
      bash $S/run_opd_code.sh >> $SCRATCH/logs/train_math17b_from${tag}_16k.log 2>&1
  local rc=$?
  echo "######## END math16k $tag rc=$rc $(date -Is) ########"
  return $rc
}
teacher_of () { case $1 in
  K4Bmath1200) echo Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200;;
  K4Bmath500)  echo Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500;;
  K4Bcode1200) echo Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200;;
  K4Bcode300)  echo Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300;;
  4b)          echo Qwen/Qwen3-4B;;
  RLCR4B)      echo resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step;;
esac; }
