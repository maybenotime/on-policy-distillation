#!/bin/bash
#SBATCH --job-name=opd-v05-3gB
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=224G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/sya225/logs/hold_venus05_%j.out
# GPU holder: work is attached later with  srun --jobid=$SLURM_JOB_ID --overlap ...
mkdir -p /localscratch/sya225
echo "holder up on $(hostname) $(date -Is)"
sleep infinity
