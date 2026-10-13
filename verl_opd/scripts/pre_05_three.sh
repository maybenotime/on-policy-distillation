#!/bin/bash
# cs-venus-05: pre-probes of three teachers vs R1-Distill-1.5B on the existing 1000 DAPO rollouts (roll_ds15b-dapo.jsonl).
#   qihoo360/Light-R1-7B-DS, nvidia/Nemotron-Research-Reasoning-Qwen-1.5B@brorl, agentica-org/DeepScaleR-1.5B-Preview
set -u; source /home/sya225/opd/scripts/topk_common.sh; S=$OPD_HOME/scripts; RT=ds15b-dapo
STU=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
echo "PRE05_START $(hostname) $(date -Is)"
# 1) downloads (Nemotron: brorl branch -> local snapshot path)
NEMO=$("$E/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
pats=["*.json","*.safetensors","*.txt","*.model","*.py","*.jinja"]
p=snapshot_download("nvidia/Nemotron-Research-Reasoning-Qwen-1.5B", revision="brorl", allow_patterns=pats)
snapshot_download("qihoo360/Light-R1-7B-DS", allow_patterns=pats); snapshot_download("agentica-org/DeepScaleR-1.5B-Preview", allow_patterns=pats)
print(p)
PY
)
NEMO=$(echo "$NEMO" | tail -1); echo "DL_DONE nemotron@brorl=$NEMO $(date -Is)"
# 2) student dump on GPU 0 (needed for overlap metrics; 19's copy is node-local)
dump 0 $RT $STU preds_student_ds15b
# 3) three teachers in parallel on GPUs 1-3
probe () {  # name model gpu
  dump $3 $RT "$2" preds_T$1
  met preds_student_ds15b preds_T$1 "pre ds1.5B vs $1"
  [ -s $F/ent_T$1_$RT.npz ] || CUDA_VISIBLE_DEVICES=$3 "$E/bin/python" $S/fork_entropy.py --model "$2" --rollouts $F/roll_$RT.jsonl --out $F/ent_T$1_$RT.npz > $L/ent_T$1_$RT.log 2>&1
  "$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_T$1_$RT.npz --label "pre ds1.5B vs $1"
  "$E/bin/python" $S/transfer_probe.py --student $D/preds_student_ds15b.npz --teacher $D/preds_T$1.npz --rollouts $F/roll_$RT.jsonl --label DS.$1 2>&1 | tail -1
  echo "PROBE_DONE $1 $(date -Is)"
}
probe Light7B  qihoo360/Light-R1-7B-DS 1 > $L/pre05_Light7B.log 2>&1 &
probe NemoBroRL "$NEMO" 2 > $L/pre05_NemoBroRL.log 2>&1 &
probe DeepScaleR agentica-org/DeepScaleR-1.5B-Preview 3 > $L/pre05_DeepScaleR.log 2>&1 &
wait
cat $L/pre05_Light7B.log $L/pre05_NemoBroRL.log $L/pre05_DeepScaleR.log | grep -E "overlap=|rho=|reach@|PROBE_DONE|Error|Traceback"
echo "PRE05_DONE $(date -Is)"
