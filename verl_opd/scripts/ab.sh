#!/bin/bash
V=/home/USER/opd/envs/verl_opd
echo "===== GPU 拓扑 ====="
nvidia-smi topo -m 2>/dev/null | head -12
echo
echo "===== nccl_new 版本 ====="
ls /localscratch/USER/nccl_new/nvidia/nccl/lib/
strings /localscratch/USER/nccl_new/nvidia/nccl/lib/libnccl.so.2 2>/dev/null | grep -oE '^2\.[0-9]+\.[0-9]+' | sort -u | tail -3
echo
echo "===== B: LD_PRELOAD nccl_new ====="
LD_PRELOAD=/localscratch/USER/nccl_new/nvidia/nccl/lib/libnccl.so.2 \
  "$V/bin/torchrun" --nproc_per_node=3 --master_port=29678 /home/USER/opd/scripts/nccl_test.py 2>&1 | grep -E '\[info\]|\[result\]|PASS|Error|error'
