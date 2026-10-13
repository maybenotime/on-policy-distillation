#!/bin/bash
# 19: 2-step smoke test of loss_mode=rk_topk_probe (student top-16 reverse KL, supervised path), Qwen3-1.7B <- Qwen3-4B, code data
source /home/sya225/opd/env.sh none; export VERL_TEACHER_KEEP_SAMPLED=1
rm -rf $SCRATCH/checkpoints/smoke_rk /tmp/ray_smoke_rk
CUDA_VISIBLE_DEVICES=0,1,2 RAY_TMPDIR=/tmp/ray_smoke_rk \
env SMOKE=1 STUDENT_MODEL=Qwen/Qwen3-1.7B TEACHER_MODEL=Qwen/Qwen3-4B DATA_DIR=/home/sya225/opd/verl_data/ot3_code30k \
    ROLLOUT_N=1 ACTOR_LR=1e-6 MAX_PROMPT_LENGTH=1024 DISTILLATION_LOSS_MODE=rk_topk_probe USE_POLICY_GRADIENT=False DISTILLATION_TOPK=64 \
    SAVE_FREQ=-1 HF_HUB_OFFLINE=1 NGPUS_PER_NODE=2 TEACHER_WORLD_SIZE=1 TEACHER_TP=1 TEACHER_GPU_MEM_UTIL=0.45 \
    CKPT_DIR=$SCRATCH/checkpoints/smoke_rk \
    bash /home/sya225/opd/scripts/run_opd_code.sh actor_rollout_ref.actor.calculate_entropy=True \
    > $SCRATCH/logs/smoke_rk.log 2>&1
echo "SMOKE_RK_DONE rc=$? $(date -Is)"
