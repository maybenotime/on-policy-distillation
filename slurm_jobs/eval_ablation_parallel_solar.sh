#!/bin/bash
#SBATCH -J eval-ablation
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=03:00:00
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

echo "=== Starting 3 evals in parallel ==="
echo "GPU0: Baseline"
echo "GPU1: S✓T✓ -> $CKPT_ST_CORRECT"
echo "GPU2: S✗T✓ -> $CKPT_S_INCORRECT"

CUDA_VISIBLE_DEVICES=0 swift eval \
    --use_hf true \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k math_500 \
    --eval_output_dir "/localscratch/sya225/eval_output/ablation_baseline_0.5B" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 8192 \
    --eval_num_proc 100 &
PID_BASELINE=$!

CUDA_VISIBLE_DEVICES=1 swift eval \
    --use_hf true \
    --model "$CKPT_ST_CORRECT" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k math_500 \
    --eval_output_dir "/localscratch/sya225/eval_output/ablation_ST_correct" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 8192 \
    --eval_num_proc 100 &
PID_ST=$!

CUDA_VISIBLE_DEVICES=2 swift eval \
    --use_hf true \
    --model "$CKPT_S_INCORRECT" \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_dataset gsm8k math_500 \
    --eval_output_dir "/localscratch/sya225/eval_output/ablation_S_incorrect_T_correct" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 8192 \
    --eval_num_proc 100 &
PID_SI=$!

echo "Waiting for all evals to finish..."
wait $PID_BASELINE && echo "Baseline done" || echo "Baseline failed"
wait $PID_ST && echo "S✓T✓ done" || echo "S✓T✓ failed"
wait $PID_SI && echo "S✗T✓ done" || echo "S✗T✓ failed"

echo "=== All evals done ==="
