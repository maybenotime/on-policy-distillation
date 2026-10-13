#!/bin/bash
# Does the student's shape alignment come from (a) its per-state logits changing, or (b) its trajectories changing?
# For each trained DS student: rho(student, JustRL) on the student's OWN new rollouts (on-policy) vs on the FIXED base-student
# rollouts (fork/roll_ds15b-dapo.jsonl, JustRL entropies already cached).  Pre-OPD reference on the fixed set: rho = 0.853.
# Node 05 GPUs 3,4,5 (idle); weights vanish with the job at 00:37, hence now.
set -u
source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; S=$OPD_HOME/scripts; F=$OPD_HOME/fork; R=$SCRATCH/rho; L=$SCRATCH/logs; H=/home/sya225/logs
export PATH="$E/bin:$PATH" TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
JR=hbx/JustRL-DeepSeek-1.5B
FIXED=$F/roll_ds15b-dapo.jsonl                       # base student's 1000 DAPO rollouts (4k/T0.7 protocol)
ENT_JR_FIXED=$F/ent_TJustRL_ds15b-dapo.npz           # JustRL's entropy on them (cached)
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
  "$E/bin/python" $S/fork_entropy.py --model $JR --rollouts $R/roll_${TAG}_own.jsonl --out $R/ent_${TAG}_own_JustRL.npz --chunk 256 > $L/pf_${TAG}_ownT.log 2>&1
  echo "[$TAG] RESULTS $(date -Is)"
  "$E/bin/python" $S/fork_metrics.py --student_ent $R/ent_${TAG}_onFIXED.npz --teacher_ent $ENT_JR_FIXED --label "$TAG  FIXED-traj  vs JustRL"
  "$E/bin/python" $S/fork_metrics.py --student_ent $R/ent_${TAG}_own_student.npz --teacher_ent $R/ent_${TAG}_own_JustRL.npz --label "$TAG  ON-POLICY   vs JustRL"
}
echo "PFO_START $(date -Is)"
run_one prefsonly_s200 $SCRATCH/checkpoints/hf_ds15b_tpJustRL_shapeStu_step200 3 > $H/pfo_prefs.out 2>&1 &
run_one shapeonly_s100 $SCRATCH/checkpoints/hf_ds15b_tpStu_shapeJustRL_step100 4 > $H/pfo_shape.out 2>&1 &
wait
echo "PFO_DONE $(date -Is)"
