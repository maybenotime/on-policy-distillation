#!/bin/bash
#SBATCH -J NLP-TokenReward
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

# Token-level reward analysis — all 5 (student, teacher) pairs
# Runs sequentially on one A40. Each pair ~1h; total ~5h within budget.
# Output: results/token_rewards_<student>_<teacher>.pkl

PROJECT_DIR="/localscratch/cta156/on-policy-distillation"
LOG_DIR="/home/cta156/NLP/Thesis/0506-chapter4/analysis_token_reward"

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

# Load conda.
source ~/miniconda/etc/profile.d/conda.sh

# Some conda activation scripts reference unset variables.
set +u
conda activate opd311
set -u

echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "Conda prefix: ${CONDA_PREFIX}"
echo "Python: $(which python)"
python -V

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

# Optional: update repo. Comment this out if you want exact reproducibility.
git status --short
git pull

mkdir -p logs results

PROMPTS="data/prompts_math_with_answers_correct_100.jsonl"

run_token_reward() {
    local STUDENT="$1"
    local TEACHER="$2"
    local STUDENT_TAG
    local TEACHER_TAG
    STUDENT_TAG=$(basename "$STUDENT")
    TEACHER_TAG=$(basename "$TEACHER")
    local OUT="results/token_rewards_${STUDENT_TAG}_${TEACHER_TAG}.pkl"

    echo ""
    echo "========================================"
    echo "Token reward: ${STUDENT_TAG} <- ${TEACHER_TAG}"
    echo "Output: ${OUT}"
    echo "========================================"

    python source/token_reward_analysis.py \
        --student_model "$STUDENT" \
        --teacher_models "$TEACHER" \
        --prompts_file "$PROMPTS" \
        --output_path "$OUT" \
        --max_new_tokens 8192 \
        --temperature 1.0 \
        --batch_size 2

    test -f "$OUT"
    echo "Saved: $OUT"
}

run_token_reward Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct
run_token_reward Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-7B-Instruct
run_token_reward Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-14B-Instruct
run_token_reward Qwen/Qwen2.5-3B-Instruct   Qwen/Qwen2.5-7B-Instruct
run_token_reward Qwen/Qwen2.5-3B-Instruct   Qwen/Qwen2.5-14B-Instruct

echo ""
echo "All token reward analyses done."
ls -lh results/token_rewards_*.pkl