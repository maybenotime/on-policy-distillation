#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=72b-mixed-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "Job $SLURM_JOB_ID running on $SLURM_NODELIST"

module load StdEnv/2023 gcc cuda/12.2 python/3.10 arrow/17.0.0
source /scratch/cta156/venvs/rethink-gap/bin/activate

HARNESS_DIR="/project/def-angelx/cta156/Rethink-Learnability-Gap/lm-evaluation-harness"
SCRATCH="/home/cta156/scratch/on-policy-distillation"

export RUN_NAME="${RUN_NAME:-chapter5-72b-mixed}"
export OUTPUT_PATH="${OUTPUT_PATH:-${SCRATCH}/eval_output/${RUN_NAME}}"
export TASKS="${TASKS:-gsm8k_cot_zeroshot,hendrycks_math_500,AMC,Olympiad,AIME}"
export CUDA_VISIBLE_DEVICES="0"

export MODEL_LIST="${MODEL_LIST:-$(cat <<'EOF'
rsft_then_opsd_3B_72bref_mixed10pct_ckpt200 /home/cta156/scratch/on-policy-distillation/output/rsft_then_opsd_3B_72bref_mixed10pct/v0-20260516-192904/checkpoint-200
rsft_then_opsd_3B_72bref_mixed30pct_ckpt200 /home/cta156/scratch/on-policy-distillation/output/rsft_then_opsd_3B_72bref_mixed30pct/v0-20260516-192904/checkpoint-200
EOF
)}"

echo "HARNESS_DIR=${HARNESS_DIR}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "TASKS=${TASKS}"

mkdir -p "$OUTPUT_PATH"
cd "$HARNESS_DIR"
bash run_lm_eval_models.sh

echo "Done."
