#!/bin/bash
# Resume Code-300 16k (paused at step 107; latest ckpt global_step_100) on node-B.
# Identical env to chain19b's run(): verl resume_mode=auto picks up
# latest_checkpointed_iteration.txt in CKPT_DIR. NGPUS_PER_NODE must stay 2 (FSDP shards).
set -u
source /home/USER/opd/env.sh none
sed -n '/^COMMON=/,/^}/p' $OPD_HOME/scripts/chain19.sh > /tmp/chain19_common.sh
source /tmp/chain19_common.sh
COMMON+=(MAX_RESPONSE_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=24576)
echo "######## RESUME K4Bcode300_16k from step $(cat $SCRATCH/checkpoints/opd_code_17b_fromK4Bcode300_16k/latest_checkpointed_iteration.txt) $(date -Is) ########"
run K4Bcode300_16k Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300 0,1,2
echo "RESUME_CODE300_DONE $(date -Is)"
