#!/bin/bash
#SBATCH -J opsd
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=11:00:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=rrg-msavva
#SBATCH --mail-user=yangshipingnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/sya225/projects/rrg-msavva/sya225/on-policy-distillation

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test_env/bin/activate

export OPSD_MODE="${OPSD_MODE:-single}"
export OPSD_TEACHER_MODEL="${OPSD_TEACHER_MODEL:-deepseek-r1}"
export OPSD_DATA_PATH="${OPSD_DATA_PATH:-/home/sya225/scratch/data/final_math_dataset_updated.jsonl}"

cd scripts

bash opsd.sh
