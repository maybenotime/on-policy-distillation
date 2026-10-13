#!/bin/bash
V=/home/sya225/opd/envs/verl_opd
cd /localscratch/sya225
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
nvidia-smi --query-gpu=index,name --format=csv,noheader
"$V/bin/torchrun" --nproc_per_node=3 --master_port=29677 /home/sya225/opd/scripts/nccl_test.py
