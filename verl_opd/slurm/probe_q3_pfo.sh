#!/bin/bash
# Qwen3 version of probe_fixed_vs_onpolicy.sh: rho(student, M500) on the FIXED base-student rollouts (fork/roll_q3i17-dapo.jsonl,
# M500 entropies cached; pre-OPD rho 0.772) vs on each student's OWN new rollouts.  Four checkpoints, one per GPU (node 19 idle).
set -u
source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; S=$OPD_HOME/scripts; F=$OPD_HOME/fork; R=$SCRATCH/rho; L=$SCRATCH/logs; H=/home/sya225/logs
export PATH="$E/bin:$PATH" TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
M5=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
FIXED=$F/roll_q3i17-dapo.jsonl; ENT_M5_FIXED=$F/ent_TK4Bmath500_q3i17-dapo.npz
mkdir -p $R
run_one () {  # tag hfdir gpu
  local TAG=$1 HF=$2 G=$3
  export CUDA_VISIBLE_DEVICES=$G
  echo "[$TAG] fixed-trajectory pass $(date -Is)"
  [ -s $R/ent_${TAG}_onFIXED.npz ] || "$E/bin/python" $S/fork_entropy.py --model $HF --rollouts $FIXED --out $R/ent_${TAG}_onFIXED.npz --chunk 256 > $L/pf_${TAG}_fixed.log 2>&1
  echo "[$TAG] on-policy rollout $(date -Is)"
  ( setsid "$E/bin/python" $S/fork_rollout.py --model $HF --data /home/sya225/opd/verl_data/dapo_math17k/train.parquet --n 1000 --seed 0 --max_tokens 4096 --out $R/roll_${TAG}_own.jsonl > $L/pf_${TAG}_roll.log 2>&1 & pid=$!
    for _ in $(seq 1 120); do grep -q '^\[fork\] wrote' $L/pf_${TAG}_roll.log 2>/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 10; done
    sleep 5; kill -9 -$pid 2>/dev/null; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
    for _ in $(seq 1 30); do [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 3000 ] 2>/dev/null && break; sleep 5; done )
  echo "[$TAG] on-policy entropies $(date -Is)"
  "$E/bin/python" $S/fork_entropy.py --model $HF --rollouts $R/roll_${TAG}_own.jsonl --out $R/ent_${TAG}_own_student.npz --chunk 256 > $L/pf_${TAG}_ownS.log 2>&1
  "$E/bin/python" $S/fork_entropy.py --model $M5 --rollouts $R/roll_${TAG}_own.jsonl --out $R/ent_${TAG}_own_M500.npz --chunk 256 > $L/pf_${TAG}_ownT.log 2>&1
  echo "[$TAG] RESULTS $(date -Is)"
  "$E/bin/python" $S/fork_metrics.py --student_ent $R/ent_${TAG}_onFIXED.npz --teacher_ent $ENT_M5_FIXED --label "$TAG  FIXED-traj  vs M500"
  "$E/bin/python" $S/fork_metrics.py --student_ent $R/ent_${TAG}_own_student.npz --teacher_ent $R/ent_${TAG}_own_M500.npz --label "$TAG  ON-POLICY   vs M500"
}
echo "PFOQ3_START $(date -Is)"
C=$SCRATCH/checkpoints
run_one q3_shapeStuFix_s100 $C/hf_q3i17_tpM500_shapeStuFix_step100 0 > $H/pfq_fix.out 2>&1 &
run_one q3_shapeA4B_s100    $C/hf_q3i17_tpM500_shapeA4B_step100    1 > $H/pfq_a4b.out 2>&1 &
run_one q3_shapeStuLive_s50 $C/hf_tpM500_shapeStu_step50            2 > $H/pfq_live.out 2>&1 &
run_one q3_M500direct_s50   $C/hf_q3i17_fromM500_step50             3 > $H/pfq_direct.out 2>&1 &
wait
cat $H/pfq_*.out | grep -E "rho=|Error|Traceback"
echo "PFOQ3_DONE $(date -Is)"
