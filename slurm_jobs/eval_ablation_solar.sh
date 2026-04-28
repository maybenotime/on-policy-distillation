#!/bin/bash
#SBATCH -J eval-ablation
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:59:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc
cd /home/sya225/on-policy-distillation
git pull
source test_env/bin/activate

export HF_HOME=/home/sya225/.cache/huggingface
export MODELSCOPE_CACHE=/home/sya225/.cache/modelscope
export TORCH_HOME=/home/sya225/.cache/torch
export VLLM_CACHE_ROOT=/home/sya225/.cache/vllm
export TMPDIR=/home/sya225/.cache/tmp
export HF_DATASETS_CACHE=/home/sya225/.cache/huggingface/datasets
mkdir -p $HF_HOME $MODELSCOPE_CACHE $TORCH_HOME $VLLM_CACHE_ROOT $TMPDIR

CKPT_ST_CORRECT="/home/sya225/on-policy-distillation/checkpoints/ST_correct_3000_checkpoint-250"
CKPT_S_INCORRECT=$(find /localscratch/sya225/output/gsm8k_S_incorrect_T_correct_3000_sQwen2.5-0.5B-Instruct_tQwen2.5-3B-Instruct -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)

echo "=== Eval 1: S✓T✓ checkpoint: $CKPT_ST_CORRECT ==="
CUDA_VISIBLE_DEVICES=0 swift eval \
    --use_hf true \
    --model "$CKPT_ST_CORRECT" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k math_500 \
    --eval_output_dir "/localscratch/sya225/eval_output/ablation_ST_correct" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 8192 \
    --eval_num_proc 100

echo "=== Eval 2: S✗T✓ checkpoint: $CKPT_S_INCORRECT ==="
CUDA_VISIBLE_DEVICES=0 swift eval \
    --use_hf true \
    --model "$CKPT_S_INCORRECT" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k math_500 \
    --eval_output_dir "/localscratch/sya225/eval_output/ablation_S_incorrect_T_correct" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 8192 \
    --eval_num_proc 100

echo "=== Eval 3: Baseline (no fine-tuning) ==="
CUDA_VISIBLE_DEVICES=0 swift eval \
    --use_hf true \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k math_500 \
    --eval_output_dir "/localscratch/sya225/eval_output/ablation_baseline_0.5B" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 8192 \
    --eval_num_proc 100

echo "=== All evals done ==="
