#!/bin/bash
V=/home/USER/opd/envs/verl_opd
cd /localscratch/USER
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
nvidia-smi --query-gpu=index,name --format=csv,noheader
"$V/bin/torchrun" --nproc_per_node=3 --master_port=29677 /home/USER/opd/scripts/nccl_test.py
