#!/bin/bash
#SBATCH -J opsd
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --nodelist=cs-venus-15
#SBATCH --time=11:00:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc

cd /home/sya225/on-policy-distillation

source test_env/bin/activate

export OPSD_MODE="${OPSD_MODE:-single}"
export OPSD_TEACHER_MODEL="${OPSD_TEACHER_MODEL:-deepseek-r1}"
export OPSD_TEACHER_MODELS="${OPSD_TEACHER_MODELS:-}"
export OPSD_DATA_PATH="${OPSD_DATA_PATH:-/home/sya225/on-policy-distillation/data/math_all_teachers_correct.jsonl}"

export HF_HOME=/home/sya225/.cache/huggingface
export MODELSCOPE_CACHE=/home/sya225/.cache/modelscope
export TORCH_HOME=/home/sya225/.cache/torch
export VLLM_CACHE_ROOT=/home/sya225/.cache/vllm
export TMPDIR=/home/sya225/.cache/tmp
export HF_DATASETS_CACHE=/home/sya225/.cache/huggingface/datasets

mkdir -p $HF_HOME $MODELSCOPE_CACHE $TORCH_HOME $VLLM_CACHE_ROOT $TMPDIR

cd scripts

bash opsd.sh
