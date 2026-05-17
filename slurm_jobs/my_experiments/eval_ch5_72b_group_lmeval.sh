#!/bin/bash
#SBATCH -J 72B-Group-Eval
#SBATCH --gres=gpu:3
#SBATCH --time=1-00:00
#SBATCH --output=/dev/null
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --nodelist=cs-venus-09
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cta156@sfu.ca

set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${RUN_NAME:-chapter5-72b-group}"
LOG_ROOT="${LOG_ROOT:-${HOME}/NLP/Thesis/0507-eval-result/${RUN_NAME}}"
mkdir -p "$LOG_ROOT"

LOG_FILE="${LOG_ROOT}/${TIMESTAMP}_job${SLURM_JOB_ID}.log"
ln -sf "$(realpath "$LOG_FILE")" "${LOG_ROOT}/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Chapter 5 72B-group lm-eval job"
echo "========================================"
echo "RUN_NAME=${RUN_NAME}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "LOG_FILE=${LOG_FILE}"

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
ulimit -Su unlimited
ulimit -Sv unlimited
ulimit -Sl unlimited

CONDA_ENV="${CONDA_ENV:-rethink-gap}"
conda activate "$CONDA_ENV"

PROJECT_ROOT="${PROJECT_ROOT:-/localscratch/cta156/Rethink-Learnability-Gap}"
HARNESS_DIR="${HARNESS_DIR:-${PROJECT_ROOT}/lm-evaluation-harness}"
cd "$HARNESS_DIR"

export RUN_NAME
export OUTPUT_PATH="${OUTPUT_PATH:-../results/${RUN_NAME}}"
export TASKS="${TASKS:-gsm8k_cot_zeroshot,hendrycks_math_500,AMC,Olympiad,AIME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

CKPT_ROOT="/3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group"

# All 4 checkpoints per model — previous OPSD runs peaked before step 200.
export MODEL_LIST="${MODEL_LIST:-$(cat <<'EOF'
sft_3B_72bref_ckpt50  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/sft_3B_72bref/v0-20260515-193641/checkpoint-50
sft_3B_72bref_ckpt100 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/sft_3B_72bref/v0-20260515-193641/checkpoint-100
sft_3B_72bref_ckpt150 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/sft_3B_72bref/v0-20260515-193641/checkpoint-150
sft_3B_72bref_ckpt200 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/sft_3B_72bref/v0-20260515-193641/checkpoint-200
opsd_3B_72bref_alpha0p0_ckpt50  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/opsd_3B_72bref_alpha0p0/v0-20260515-193752/checkpoint-50
opsd_3B_72bref_alpha0p0_ckpt100 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/opsd_3B_72bref_alpha0p0/v0-20260515-193752/checkpoint-100
opsd_3B_72bref_alpha0p0_ckpt150 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/opsd_3B_72bref_alpha0p0/v0-20260515-193752/checkpoint-150
opsd_3B_72bref_alpha0p0_ckpt200 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/opsd_3B_72bref_alpha0p0/v0-20260515-193752/checkpoint-200
opsd_3B_72bref_alpha0p5_ckpt50  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/opsd_3B_72bref_alpha0p5/v0-20260515-194113/checkpoint-50
opsd_3B_72bref_alpha0p5_ckpt100 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/opsd_3B_72bref_alpha0p5/v0-20260515-194113/checkpoint-100
opsd_3B_72bref_alpha0p5_ckpt150 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/opsd_3B_72bref_alpha0p5/v0-20260515-194113/checkpoint-150
opsd_3B_72bref_alpha0p5_ckpt200 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/opsd_3B_72bref_alpha0p5/v0-20260515-194113/checkpoint-200
rsft_then_opsd_3B_72bref_ckpt50  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/rsft_then_opsd_3B_72bref/v0-20260515-194046/checkpoint-50
rsft_then_opsd_3B_72bref_ckpt100 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/rsft_then_opsd_3B_72bref/v0-20260515-194046/checkpoint-100
rsft_then_opsd_3B_72bref_ckpt150 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/rsft_then_opsd_3B_72bref/v0-20260515-194046/checkpoint-150
rsft_then_opsd_3B_72bref_ckpt200 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/72b-group/rsft_then_opsd_3B_72bref/v0-20260515-194046/checkpoint-200
EOF
)}"

echo "HARNESS_DIR=${HARNESS_DIR}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "TASKS=${TASKS}"
echo "CONDA_ENV=${CONDA_ENV}"
echo

bash run_lm_eval_models.sh

RESULTS_ABS="$(realpath "$OUTPUT_PATH")"
PACK_RESULTS="${PACK_RESULTS:-1}"
if [[ "$PACK_RESULTS" == "1" ]]; then
    ARCHIVE_DIR="${ARCHIVE_DIR:-${LOG_ROOT}/archives}"
    mkdir -p "$ARCHIVE_DIR"
    ARCHIVE_PATH="${ARCHIVE_DIR}/${RUN_NAME}_${SLURM_JOB_ID}_${TIMESTAMP}.tar.gz"

    echo
    echo "Packing results"
    echo "  source:  ${RESULTS_ABS}"
    echo "  archive: ${ARCHIVE_PATH}"
    tar -czf "$ARCHIVE_PATH" -C "$(dirname "$RESULTS_ABS")" "$(basename "$RESULTS_ABS")"

    if [[ -n "${TRANSFER_TARGET:-}" ]]; then
        echo
        echo "Transferring archive to ${TRANSFER_TARGET}"
        rsync -av "$ARCHIVE_PATH" "$TRANSFER_TARGET"
    else
        if [[ -z "${SCP_HOST:-}" ]]; then
            SCP_HOST="$(hostname -f 2>/dev/null || hostname)"
        fi
        echo
        echo "Archive ready. Pull it with:"
        echo "scp ${USER}@${SCP_HOST}:${ARCHIVE_PATH} ~/Downloads/"
    fi
else
    echo
    echo "PACK_RESULTS=0, skipped result archive."
fi

echo
echo "Evaluation, packing, and optional transfer completed."
