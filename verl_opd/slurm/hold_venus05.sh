#!/bin/bash
#SBATCH --job-name=opd-v05-6g
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=96
#SBATCH --mem=448G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/sya225/logs/hold_venus05_%j.out
# GPU holder: work is attached later with  srun --jobid=$SLURM_JOB_ID --overlap ...
mkdir -p /localscratch/sya225
echo "holder up on $(hostname) $(date -Is)"
sleep infinity
