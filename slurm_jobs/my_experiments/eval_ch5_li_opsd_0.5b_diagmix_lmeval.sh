#!/bin/bash
#SBATCH -J LiOPSD-Mix-Eval
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00
#SBATCH --output=/dev/null
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --nodelist=cs-venus-07
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cta156@sfu.ca

set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${RUN_NAME:-chapter5-li-opsd-5k-0.5b-diagmix}"
LOG_ROOT="${LOG_ROOT:-${HOME}/NLP/Thesis/0507-eval-result/${RUN_NAME}}"
mkdir -p "$LOG_ROOT"

LOG_FILE="${LOG_ROOT}/${TIMESTAMP}_job${SLURM_JOB_ID}.log"
ln -sf "$(realpath "$LOG_FILE")" "${LOG_ROOT}/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Chapter 5 LI-OPSD 5k 0.5B diagmix lm-eval job"
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export MODEL_LIST="${MODEL_LIST:-$(cat <<'EOF'
li_opsd_0.5B_diagmix50_50_ratio50_ckpt25  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-25
li_opsd_0.5B_diagmix50_50_ratio50_ckpt50  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-50
li_opsd_0.5B_diagmix50_50_ratio50_ckpt75  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-75
li_opsd_0.5B_diagmix50_50_ratio50_ckpt100 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-100
li_opsd_0.5B_diagmix50_50_ratio50_ckpt125 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-125
li_opsd_0.5B_diagmix50_50_ratio50_ckpt150 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-150
li_opsd_0.5B_diagmix50_50_ratio50_ckpt175 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-175
li_opsd_0.5B_diagmix50_50_ratio50_ckpt200 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-200
li_opsd_0.5B_diagmix50_50_ratio50_ckpt225 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-225
li_opsd_0.5B_diagmix50_50_ratio50_ckpt250 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-250
li_opsd_0.5B_diagmix50_50_ratio50_ckpt275 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-275
li_opsd_0.5B_diagmix50_50_ratio50_ckpt300 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-300
li_opsd_0.5B_diagmix50_50_ratio50_ckpt320 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix50_50_ratio50/v0-20260518-060720/checkpoint-320
li_opsd_0.5B_diagmix70_30_ratio50_ckpt25  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-25
li_opsd_0.5B_diagmix70_30_ratio50_ckpt50  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-50
li_opsd_0.5B_diagmix70_30_ratio50_ckpt75  /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-75
li_opsd_0.5B_diagmix70_30_ratio50_ckpt100 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-100
li_opsd_0.5B_diagmix70_30_ratio50_ckpt125 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-125
li_opsd_0.5B_diagmix70_30_ratio50_ckpt150 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-150
li_opsd_0.5B_diagmix70_30_ratio50_ckpt175 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-175
li_opsd_0.5B_diagmix70_30_ratio50_ckpt200 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-200
li_opsd_0.5B_diagmix70_30_ratio50_ckpt225 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-225
li_opsd_0.5B_diagmix70_30_ratio50_ckpt250 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-250
li_opsd_0.5B_diagmix70_30_ratio50_ckpt275 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-275
li_opsd_0.5B_diagmix70_30_ratio50_ckpt300 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-300
li_opsd_0.5B_diagmix70_30_ratio50_ckpt320 /3dlg-jupiter-project/rethink_gap/model_checkpoints_chapter5/li-opsd-5k/li_opsd_0.5B_diagmix70_30_ratio50/v0-20260518-060720/checkpoint-320
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
