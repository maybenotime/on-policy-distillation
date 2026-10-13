#!/bin/bash
#SBATCH --job-name=opd-v05-2g
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/sya225/logs/hold_venus05_%j.out
mkdir -p /localscratch/sya225
echo "holder up on $(hostname) $(date -Is)"
sleep infinity
