#!/bin/bash
#SBATCH --job-name=opd-v05-cpu
#SBATCH --partition=3dlg-hcvc-lab-long
#SBATCH --nodelist=cs-venus-05
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6-00:00:00
#SBATCH --output=/home/sya225/logs/hold_venus05_%j.out
# CPU-only holder: keeps /localscratch/sya225 on cs-venus-05 reachable via  srun --jobid=$SLURM_JOB_ID --overlap -c 4 --ntasks=1 ...
echo "cpu holder up on $(hostname) $(date -Is)"
sleep infinity
