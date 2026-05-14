#!/bin/bash
#SBATCH --account=def-angelx_gpu
#SBATCH --job-name=exp6-7B-14B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --time=15:00:00
#SBATCH --output=logs/rethink_gap/%x-%j.out
#SBATCH --error=logs/rethink_gap/%x-%j.err
#SBATCH --mail-user=cta156@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/cta156/scratch/on-policy-distillation
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test/bin/activate

save_checkpoint() {
    OUTPUT_BASE="/home/cta156/scratch/on-policy-distillation/output/student_Qwen2.5-7B-Instruct_teacher_Qwen2.5-14B-Instruct_data5000"
    CHECKPOINT=$(find "$OUTPUT_BASE" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "/home/cta156/scratch/on-policy-distillation/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint saved by trap: $CHECKPOINT"
    else
        echo "No checkpoint found at exit" >&2
    fi
}
trap save_checkpoint EXIT

cd scripts

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash rollout.sh Qwen/Qwen2.5-7B-Instruct &
ROLLOUT_PID=$!
sleep 30

echo "Start training"
CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 bash on-policy-distill.sh \
	Qwen/Qwen2.5-7B-Instruct \
	Qwen/Qwen2.5-14B-Instruct \
	5000 \
	../data/prompts_math_swift.jsonl \
	zero3
echo "Training finished"

kill $ROLLOUT_PID
wait $ROLLOUT_PID 2>/dev/null
