#!/bin/bash
#SBATCH -J NLP-Distill
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/dev/null
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodelist=cs-venus-14
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cta156@sfu.ca

set -euo pipefail

PROJECT_DIR="/localscratch/cta156/on-policy-distillation"
LOG_DIR="/home/cta156/NLP/Thesis/0506-chapter4/analysis_ood"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_job${SLURM_JOB_ID}.log"
ln -sf "$(realpath "$LOG_FILE")" "${LOG_DIR}/latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "WorkDir before cd: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "========================================"

nvidia-smi || true

# Load conda properly.
source ~/miniconda/etc/profile.d/conda.sh
set +u
conda activate opd311
set -u

echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "Conda prefix: ${CONDA_PREFIX}"
echo "Python: $(which python)"
python -V

# CUDA paths from conda env.
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=$CUDA_HOME"
which nvcc || true
nvcc --version || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

cd "$PROJECT_DIR"
mkdir -p logs results

PROMPTS="data/prompts_math_with_answers_correct_100.jsonl"
NUM_ROLLOUTS=10
N_POSITIONS=10
UTIL=0.8
BS=2

run_ood() {
    local STUDENT="$1"
    local TEACHER="$2"
    local STUDENT_TAG
    local TEACHER_TAG
    STUDENT_TAG=$(basename "$STUDENT")
    TEACHER_TAG=$(basename "$TEACHER")
    local OUT="results/ood_${STUDENT_TAG}_${TEACHER_TAG}.pkl"

    echo ""
    echo "========================================"
    echo "OOD: ${STUDENT_TAG} <- ${TEACHER_TAG}"
    echo "Output: ${OUT}"
    echo "========================================"

    python source/ood_analysis.py \
        --student_model "$STUDENT" \
        --teacher_model "$TEACHER" \
        --prompts_file "$PROMPTS" \
        --output_path "$OUT" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --num_sample_positions "$N_POSITIONS" \
        --batch_size "$BS" \
        --use_vllm \
        --vllm_gpu_memory_utilization "$UTIL"

    test -f "$OUT"
    echo "Saved: $OUT"
}

run_ood Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct
run_ood Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-7B-Instruct
run_ood Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-14B-Instruct
run_ood Qwen/Qwen2.5-3B-Instruct   Qwen/Qwen2.5-7B-Instruct
run_ood Qwen/Qwen2.5-3B-Instruct   Qwen/Qwen2.5-14B-Instruct

echo ""
echo "All OOD analyses done."
ls -lh results/ood_*.pkl