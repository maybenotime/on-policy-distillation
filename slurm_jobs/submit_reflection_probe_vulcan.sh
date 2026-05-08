#!/bin/bash -l
#SBATCH -J refl-probe
#SBATCH --account=aip-angelx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Teacher reflection probe — single L40S 48GB.
# Args:
#   $1 = input pkl path (existing OOD analysis output)
#   $2 = teacher model
#   $3 = output pkl path
#   $4 = enable_thinking (true|false|none, default false for Qwen3)
#   $5 = num_samples (default 20)

cd /project/aip-angelx/sya225/on-policy-distillation

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

source test/bin/activate

mkdir -p logs results

export HF_HOME=/scratch/sya225/hf_cache
export HF_DATASETS_CACHE=/scratch/sya225/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/sya225/hf_cache
export TMPDIR=/scratch/sya225/hf_cache/tmp
mkdir -p "$TMPDIR"

INPUT_PKL=${1:-"results/ood_math_100_Qwen3-1.7B_Qwen3-8B.pkl"}
TEACHER_MODEL=${2:-"Qwen/Qwen3-8B"}
OUTPUT_PATH=${3:-"results/reflection_probe_$(basename "$INPUT_PKL" .pkl).pkl"}
ENABLE_THINKING=${4:-"false"}
NUM_SAMPLES=${5:-20}

echo "Reflection probe: input=$INPUT_PKL teacher=$TEACHER_MODEL"
echo "                  output=$OUTPUT_PATH enable_thinking=$ENABLE_THINKING"

python source/teacher_reflection_probe.py \
    --input_pkl "$INPUT_PKL" \
    --teacher_model "$TEACHER_MODEL" \
    --output_path "$OUTPUT_PATH" \
    --mode both \
    --filter incorrect \
    --num_samples "$NUM_SAMPLES" \
    --max_new_tokens 2048 \
    --temperature 1.0 \
    --vllm_gpu_memory_utilization 0.85 \
    --enable_thinking "$ENABLE_THINKING"
