#!/bin/bash
#SBATCH --job-name=opd-v19-4g
#SBATCH --partition=3dlg-hcvc-lab-short
#SBATCH --nodelist=cs-venus-19
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/sya225/logs/hold_venus19_%j.out
# GPU holder: work is attached later with  srun --jobid=$SLURM_JOB_ID --overlap ...
mkdir -p /localscratch/sya225
echo "holder up on $(hostname) $(date -Is)"
sleep infinity
