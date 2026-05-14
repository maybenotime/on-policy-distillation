#!/bin/bash
#SBATCH -J on-policy-t3b-s0.5b-5000
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclusive
#SBATCH --nodelist=cs-venus-15
#SBATCH --time=11:59:00
#SBATCH --output=logs/%N-%j.out

source ~/.bashrc
cd /home/sya225/on-policy-distillation
source test_env/bin/activate

save_checkpoint() {
    OUTPUT_BASE="/localscratch/sya225/output/student_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-3B-Instruct_data5000"
    CHECKPOINT=$(find "$OUTPUT_BASE" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
    if [ -n "$CHECKPOINT" ]; then
        echo "$CHECKPOINT" > "/home/sya225/on-policy-distillation/checkpoint_path_${SLURM_JOB_ID}.txt"
        echo "Checkpoint saved by trap: $CHECKPOINT"
    else
        echo "No checkpoint found at exit" >&2
    fi
}
trap save_checkpoint EXIT

cd scripts

CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash rollout.sh Qwen/Qwen2.5-0.5B-Instruct &
ROLLOUT_PID=$!
sleep 30
echo "Start training"
CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 bash on-policy-distill.sh Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct 5000
echo "Training finished"
kill $ROLLOUT_PID
wait $ROLLOUT_PID 2>/dev/null
