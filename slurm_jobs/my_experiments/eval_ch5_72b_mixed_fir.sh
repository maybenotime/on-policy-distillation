#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=72b-mixed-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "Job $SLURM_JOB_ID running on $SLURM_NODELIST"

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source /scratch/cta156/on-policy-distillation/test/bin/activate

HARNESS_DIR="/project/def-angelx/cta156/Rethink-Learnability-Gap/lm-evaluation-harness"
SCRATCH="/home/cta156/scratch/on-policy-distillation"
OUTPUT_BASE="${SCRATCH}/eval_output/chapter5-72b-mixed"
TASKS="gsm8k_cot_zeroshot,hendrycks_math_500,AMC,Olympiad,AIME"

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$HARNESS_DIR"

# Use ordered array to ensure consistent eval order
LABELS=(
    "rsft_then_opsd_3B_72bref_mixed10pct_ckpt200"
    "rsft_then_opsd_3B_72bref_mixed30pct_ckpt200"
)
PATHS=(
    "${SCRATCH}/output/rsft_then_opsd_3B_72bref_mixed10pct/v0-20260516-192904/checkpoint-200"
    "${SCRATCH}/output/rsft_then_opsd_3B_72bref_mixed30pct/v0-20260516-192904/checkpoint-200"
)

for i in "${!LABELS[@]}"; do
    LABEL="${LABELS[$i]}"
    MODEL_PATH="${PATHS[$i]}"
    OUT_DIR="${OUTPUT_BASE}/${LABEL}"
    mkdir -p "$OUT_DIR"

    echo "========================================"
    echo "Evaluating: ${LABEL}"
    echo "Path:       ${MODEL_PATH}"
    echo "Output:     ${OUT_DIR}"
    echo "========================================"

    accelerate launch --num_processes 4 \
        -m lm_eval --model hf \
        --model_args pretrained="${MODEL_PATH}",dtype=bfloat16 \
        --tasks "$TASKS" \
        --batch_size auto \
        --gen_kwargs do_sample=false,temperature=0,max_gen_toks=8192 \
        --log_samples \
        --trust_remote_code \
        --output_path "$OUT_DIR" \
        --apply_chat_template

    echo "Done: ${LABEL}"
done

echo "========================================"
echo "All evals complete. Packing results..."
echo "========================================"

ARCHIVE="${SCRATCH}/eval_output/chapter5-72b-mixed_${SLURM_JOB_ID}.tar.gz"
tar -czf "$ARCHIVE" -C "${SCRATCH}/eval_output" "chapter5-72b-mixed"

echo "Archive: ${ARCHIVE}"
echo "Pull with:"
echo "  scp ${USER}@$(hostname -f):${ARCHIVE} ~/Downloads/"
