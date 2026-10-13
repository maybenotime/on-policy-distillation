#!/bin/bash
#SBATCH --job-name=opd-v05-1g
#SBATCH --partition=3dlg-hcvc-lab-debug
#SBATCH --nodelist=cs-venus-05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sya225/logs/hold_venus05_%j.out
echo "1-gpu holder up on $(hostname) $(date -Is)"; nvidia-smi -L
sleep infinity
