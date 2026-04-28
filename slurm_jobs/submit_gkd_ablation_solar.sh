#!/bin/bash
#SBATCH -J gkd-ablation
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclusive
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc
cd /home/sya225/on-policy-distillation
git pull
source test_env/bin/activate

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-3B-Instruct"}
DATASET=${3:-"data/gsm8k_ST_correct_3000.jsonl"}
DATA_SIZE=${4:-3000}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")
DATASET_TAG=$(basename "$DATASET" .jsonl)

OUTPUT_BASE="/localscratch/sya225/output/${DATASET_TAG}_s${STUDENT_TAG}_t${TEACHER_TAG}"

save_checkpoint() {
    CHECKPOINT=$(find "$OUTPUT_BASE" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "/home/sya225/on-policy-distillation/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint saved by trap: $CHECKPOINT"
    else
        echo "No checkpoint found at exit" >&2
    fi
}
trap save_checkpoint EXIT

cd scripts

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash rollout.sh "$STUDENT_MODEL" &
ROLLOUT_PID=$!
sleep 30
echo "Start training: dataset=$DATASET, size=$DATA_SIZE"

export HF_HOME=/home/sya225/.cache/huggingface
export MODELSCOPE_CACHE=/home/sya225/.cache/modelscope
export TORCH_HOME=/home/sya225/.cache/torch
export VLLM_CACHE_ROOT=/home/sya225/.cache/vllm
export TMPDIR=/home/sya225/.cache/tmp
export HF_DATASETS_CACHE=/home/sya225/.cache/huggingface/datasets

mkdir -p $HF_HOME $MODELSCOPE_CACHE $TORCH_HOME $VLLM_CACHE_ROOT $TMPDIR

HF_TOKEN="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"
huggingface-cli login --token $HF_TOKEN

export WANDB_API_KEY="8b34bbf9238dfc1c55c378dc0dea31921d5f420b"
export WANDB_PROJECT="on_policy_distill"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type gkd \
    --use_hf true \
    --model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --train_type full \
    --dataset "../${DATASET}#${DATA_SIZE}" \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 8192 \
    --max_completion_length 4096 \
    --report_to wandb \
    --output_dir "$OUTPUT_BASE" \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 16 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --padding_free False \
    --use_logits_to_keep false \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000

echo "Training finished"
kill $ROLLOUT_PID
wait $ROLLOUT_PID 2>/dev/null
