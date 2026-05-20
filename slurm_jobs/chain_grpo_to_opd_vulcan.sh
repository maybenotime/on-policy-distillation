#!/bin/bash -l
#SBATCH -J chain-g2o
#SBATCH --account=aip-angelx
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j-chain.out
#SBATCH --mail-user=yangshipngnlp@gmail.com
#SBATCH --mail-type=FAIL

# Tiny chain job: after a GRPO train finishes, find its checkpoint and
# submit an OPD train using that checkpoint as teacher.
#
# Submit with --dependency=afterany:<GRPO_JID> so it fires whether GRPO
# completed normally or hit the time limit (we just need the saved ckpt).
#
# Args:
#   $1 = GRPO_JID         the GRPO train job id this chain depends on
#   $2 = STUDENT_MODEL    HF id or local path of the student for OPD
#   $3 = MODEL_TAG        model basename used in GRPO output_dir (e.g. Qwen2.5-0.5B-Instruct)
#   $4 = DATA_SIZE        OPD data size (default 10000)
#   $5 = NUM_EPOCHS       OPD epochs (default 1)
#
# Env (optional):
#   GRPO_DATA_SIZE        GRPO output_dir uses this size in the name (default 10000)
#   GRPO_NUM_EPOCHS       GRPO output_dir epochs in name (default 1)
#   OUTPUT_BASE_DIR       defaults to /scratch/sya225/checkpoints

set -e

REPO_DIR=/project/aip-angelx/sya225/on-policy-distillation
cd "$REPO_DIR"

GRPO_JID=${1:?GRPO jid required}
STUDENT_MODEL=${2:?student model required}
MODEL_TAG=${3:?GRPO model tag required (basename of GRPO model)}
DATA_SIZE=${4:-10000}
NUM_EPOCHS=${5:-1}

GRPO_DATA_SIZE="${GRPO_DATA_SIZE:-10000}"
GRPO_NUM_EPOCHS="${GRPO_NUM_EPOCHS:-1}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/scratch/sya225/checkpoints}"

# Try to find the GRPO checkpoint two ways:
#   1. checkpoint_path_<GRPO_JID>.txt written by trap (preferred)
#   2. scan the canonical OUTPUT_DIR for the latest checkpoint-* dir
CKPT_FILE="${REPO_DIR}/checkpoint_path_${GRPO_JID}.txt"
GRPO_OUTPUT_DIR="${OUTPUT_BASE_DIR}/grpo_${MODEL_TAG}_epochs${GRPO_NUM_EPOCHS}_data${GRPO_DATA_SIZE}"

CKPT=""
if [ -f "$CKPT_FILE" ]; then
    CKPT=$(cat "$CKPT_FILE" | tr -d '[:space:]')
    echo "[chain] read ckpt from trap file: $CKPT"
fi

if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    CKPT=$(find "$GRPO_OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    echo "[chain] discovered ckpt by scan: $CKPT"
fi

if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
    echo "[chain] ERROR: no GRPO checkpoint found for jid=$GRPO_JID model=$MODEL_TAG" >&2
    echo "[chain]   tried trap file: $CKPT_FILE" >&2
    echo "[chain]   tried scan: $GRPO_OUTPUT_DIR/checkpoint-*" >&2
    exit 1
fi

# Tag the OPD output so it's easy to distinguish from baseline OPD runs.
export OUTPUT_TAG_SUFFIX="data${DATA_SIZE}_grpo${GRPO_JID}"

echo "[chain] launching OPD:"
echo "[chain]   student=$STUDENT_MODEL"
echo "[chain]   teacher=$CKPT (GRPO jid=$GRPO_JID)"
echo "[chain]   data=$DATA_SIZE epochs=$NUM_EPOCHS suffix=$OUTPUT_TAG_SUFFIX"

OPD_JID=$(sbatch --parsable \
    slurm_jobs/submit_opd_colocate_4node_vulcan.sh \
    "$STUDENT_MODEL" "$CKPT" "$DATA_SIZE" "$NUM_EPOCHS")

echo "[chain] submitted OPD job $OPD_JID"
