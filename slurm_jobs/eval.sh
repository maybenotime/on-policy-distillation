#!/bin/bash
#SBATCH -J on-policy-teacher14b-student0.5b-20000-eval
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:59:00
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=288G
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=def-angelx
#SBATCH --mail-user=yangshipingnlp@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/sya225/scratch/on-policy-distillation

module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv

source test/bin/activate

cd scripts

bash eval_math_reasoning.sh /home/sya225/scratch/on-policy-distillation/output/student_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-14B-Instruct/v0-20260316-121605/checkpoint-834 
