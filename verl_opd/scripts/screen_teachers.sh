#!/bin/bash
# Pre-OPD teacher screening: score each candidate teacher's per-position entropy
# on the SAME fixed Qwen3-1.7B rollouts (roll_Qwen3-1.7B.jsonl), then rho /
# ForkOverlap vs ent_student_Qwen3-1.7B.npz. HF forward only, one GPU each.
# Three workers pull from a shared queue; GPUs 3,4,5 only (0-2 are training).
E=/home/sya225/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache TOKENIZERS_PARALLELISM=false
D=/home/sya225/opd/fork
R=$D/roll_Qwen3-1.7B.jsonl
Q=/localscratch/sya225/screen_queue.txt
LOCK=/localscratch/sya225/screen_queue.lock
cat > $Q <<'LIST'
K4Bcode300   Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300
K4Bcode1200  Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200
K4Bmath500   Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
K4Bmath1200  Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200
RLCR4B       resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step
X8Bmath      xiaoyuchen1/Qwen3-8B-GRPO-MATH
C8Bcode      caiyuchen/Qwen3-8B-Code-RL
C4Bmath      caiyuchen/Qwen3-4B-Non-Thinking-Math-RL
P4Bcode      polaris-73/Qwen3-4B-RLVR-Code-step-500
I4B2507      Qwen/Qwen3-4B-Instruct-2507
LIST

worker () {   # gpu
  while true; do
    line=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && break
    set -- $line; tag=$1; model=$2
    out=$D/ent_T${tag}_Qwen3-1.7B.npz
    if [ -s "$out" ]; then echo "[$tag] exists, skip"; continue; fi
    echo "######## [$tag] $model gpu=$GPU start $(date -Is) ########"
    CUDA_VISIBLE_DEVICES=$GPU "$E/bin/python" /home/sya225/opd/scripts/fork_entropy.py \
        --model "$model" --rollouts "$R" --out "$out" \
        > /localscratch/sya225/logs/screen_${tag}.log 2>&1
    rc=$?
    echo "######## [$tag] rc=$rc $(date -Is) ########"
    [ $rc -eq 0 ] && "$E/bin/python" /home/sya225/opd/scripts/fork_metrics.py \
        --student_ent $D/ent_student_Qwen3-1.7B.npz --teacher_ent "$out" --label "pre 1.7B vs $tag" \
        | tee /localscratch/sya225/logs/screen_${tag}.metrics
  done
}
for g in 3 4 5; do GPU=$g worker > /localscratch/sya225/logs/screen_worker_$g.log 2>&1 & done
wait
echo "SCREEN_DONE $(date -Is)"
