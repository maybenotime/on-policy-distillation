#!/bin/bash
# 19 (job 270249): 2-step smoke test of loss_mode=k1_transplant, A = Qwen3-4B, B = Qwen3-4B-Math-Step500, GPUs 0,1,2
source /home/USER/opd/env.sh none
rm -rf $SCRATCH/checkpoints/opd_smoke_tp /tmp/ray_smoke_tp; rm -f $SCRATCH/logs/train_smoke_tp.log
SMOKE=1 SAVE_FREQ=-1 bash /home/USER/opd/scripts/chain19_tp.sh smoke_tp shape Qwen/Qwen3-4B Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 0,1,2 /tmp/ray_smoke_tp
echo "SMOKE_TP_DONE $(date -Is)"
grep -a " step:" $SCRATCH/logs/train_smoke_tp.log | grep -o "step:[0-9]* \|tp/[a-zA-Z_0-9]*:[^ ]*\|probe/rho:[^ ]*\|probe/student_entropy:[^ ]*\|distillation/abs_loss:[^ ]*\|timing_s/step:[^ ]*" | sed "s/np.float64(//; s/)//" | tr "\n" " "; echo
