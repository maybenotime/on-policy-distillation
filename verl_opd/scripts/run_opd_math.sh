#!/usr/bin/env bash
# On-policy distillation with verl on cs-venus-05 (3x RTX PRO 6000 Blackwell, 96GB).
#
# GPU split: 2 GPUs actor(train+vLLM rollout) + 1 GPU teacher(vLLM).
# Adapted from verl v0.9.0 examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh
#
# Usage:
#   source /home/sya225/opd/env.sh
#   bash /home/sya225/opd/scripts/run_opd_math.sh                 # full run
#   SMOKE=1 bash /home/sya225/opd/scripts/run_opd_math.sh         # 2 steps, tiny batch
set -xeuo pipefail

source /home/sya225/opd/env.sh

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}

NNODES=1
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}          # actor + rollout
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}  # teacher vLLM
rollout_tp=${ROLLOUT_TP:-1}
teacher_tp=${TEACHER_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.35}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.6}

# k3 = reverse-KL estimator (matches the OPD objective in your ms-swift GKD setup).
# use_policy_gradient=False => pure distillation, no task reward mixed in.
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k3}
use_policy_gradient=${USE_POLICY_GRADIENT:-False}
distillation_topk=${DISTILLATION_TOPK:-64}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
actor_lr=${ACTOR_LR:-1e-6}
total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:--1}

DATA_DIR=${DATA_DIR:-/home/sya225/opd/verl_data/gsm8k}
CKPT_DIR=${CKPT_DIR:-/localscratch/sya225/checkpoints/verl_opd}
LOGGER=${LOGGER:-'["console"]'}

EXTRA_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
    train_batch_size=8; ppo_mini_batch_size=8
    max_response_length=256; ppo_max_token_len_per_gpu=4096
    EXTRA_ARGS+=(trainer.total_training_steps=2)
fi

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

# Ray leaves a stale raylet socket behind after a killed run; next run then dies
# with "Failed to register worker to Raylet". Always clear it.
rm -rf /tmp/ray
mkdir -p "$CKPT_DIR"
# Run from node-local disk: os.getcwd() inside Ray workers intermittently
# raises ENOENT when cwd is on BeeGFS /home (seen on cs-venus-19).
cd "$CKPT_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['$DATA_DIR/train.parquet']" \
    data.val_files="['$DATA_DIR/test.parquet']" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.dataloader_num_workers=${DL_WORKERS:-0} \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.model.use_remove_padding=${REMOVE_PADDING:-True} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL:-flash_attention_2} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-True} \
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    trainer.use_v1=False \
    trainer.balance_batch=True \
    trainer.logger="$LOGGER" \
    trainer.project_name=verl_opd_math \
    trainer.experiment_name="$(basename $STUDENT_MODEL)_from_$(basename $TEACHER_MODEL)" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="$CKPT_DIR" \
    distillation.enabled=True \
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
    distillation.nnodes=${NNODES} \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util} \
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens} \
    distillation.distillation_loss.loss_mode=${distillation_loss_mode} \
    distillation.distillation_loss.topk=${distillation_topk} \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient} \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    "${EXTRA_ARGS[@]}" "$@"
