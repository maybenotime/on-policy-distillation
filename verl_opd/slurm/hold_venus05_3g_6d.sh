#!/bin/bash
#SBATCH --job-name=opd-v05-3g6d
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/sya225/logs/hold_venus05_%j.out
# GPU holder (3 cards): work is attached later with  srun --jobid=$SLURM_JOB_ID --overlap -c 48 --ntasks=1 ...
mkdir -p /localscratch/sya225
echo "holder up on $(hostname) $(date -Is)"
nvidia-smi -L
sleep infinity
