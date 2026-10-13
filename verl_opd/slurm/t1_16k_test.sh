#!/bin/bash
# cs-venus-05: offline probe under the ONLINE protocol (T=1.0, 16k) — 300 DAPO prompts, student DS-1.5B; all six DS teachers.
source /home/sya225/opd/scripts/topk_common.sh; S=$OPD_HOME/scripts; export VLLM_LOGGING_LEVEL=WARNING
STU=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; RT=ds15b-dapo-T1L16k; R=$F/roll_$RT.jsonl; DATA=/home/sya225/opd/verl_data/dapo_math17k/train.parquet
NEMO=$(ls -d /localscratch/sya225/hf_cache/hub/models--nvidia--Nemotron-Research-Reasoning-Qwen-1.5B/snapshots/*/ | head -1)
echo "T1L16K_START $(date -Is)"
if [ ! -s $R ]; then
  CUDA_VISIBLE_DEVICES=0 "$E/bin/python" $S/reach_rollout.py --model $STU --family t1 --data $DATA --n 300 --seed 0 --max_tokens 16384 --out $R > $L/roll_$RT.log 2>&1 &
  PID=$!; while kill -0 $PID 2>/dev/null; do sleep 15; [ -s $R ] && [ "$(wc -l < $R)" -ge 300 ] && { sleep 20; kill -9 $PID 2>/dev/null; }; done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); echo "$c" | grep -q EngineCore && [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -9 $p; done
fi
grep "\[reach\]" $L/roll_$RT.log
ent () { CUDA_VISIBLE_DEVICES=$3 "$E/bin/python" $S/fork_entropy.py --model "$2" --rollouts $R --out $F/ent_$1_$RT.npz > $L/ent_$1_$RT.log 2>&1; }
ent student $STU 0 & ent TJustRL hbx/JustRL-DeepSeek-1.5B 1 & ent TSky7B Skywork/Skywork-OR1-Math-7B 2 & ent TDS7B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 3 &
ent TLight7B qihoo360/Light-R1-7B-DS 4 & ent TNemoBroRL "$NEMO" 5 & wait
ent TDeepScaleR agentica-org/DeepScaleR-1.5B-Preview 1
for t in TJustRL:JustRL TSky7B:Skywork TDS7B:R1-7B TLight7B:Light-R1-7B TNemoBroRL:Nemotron-BroRL TDeepScaleR:DeepScaleR; do "$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_${t%%:*}_$RT.npz --label "T=1.0 16k: ds1.5B vs ${t##*:}"; done
echo "T1L16K_DONE $(date -Is)"
