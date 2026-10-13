#!/bin/bash
# cs-venus-05: does sampling temperature explain the online/offline rho gap?  300 student rollouts at T=1.0 (4k), then
# student / Skywork / JustRL / R1-7B entropies and rho.
source /home/sya225/opd/scripts/topk_common.sh; S=$OPD_HOME/scripts; export VLLM_LOGGING_LEVEL=WARNING
STU=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; RT=ds15b-dapo-L16k; R=$F/roll_$RT.jsonl; DATA=/home/sya225/opd/verl_data/dapo_math17k/train.parquet
echo "L16K_START $(date -Is)"
if [ ! -s $R ]; then
  CUDA_VISIBLE_DEVICES=4 "$E/bin/python" $S/reach_rollout.py --model $STU --family q3off --data $DATA --n 300 --seed 0 --max_tokens 16384 --out $R > $L/roll_$RT.log 2>&1 &
  PID=$!; while kill -0 $PID 2>/dev/null; do sleep 15; [ -s $R ] && [ "$(wc -l < $R)" -ge 300 ] && { sleep 20; kill -9 $PID 2>/dev/null; }; done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); echo "$c" | grep -q EngineCore && [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -9 $p; done
fi
grep "\[reach\]" $L/roll_$RT.log
ent () { CUDA_VISIBLE_DEVICES=$3 "$E/bin/python" $S/fork_entropy.py --model "$2" --rollouts $R --out $F/ent_$1_$RT.npz > $L/ent_$1_$RT.log 2>&1; }
ent student $STU 4 & ent TSky7B Skywork/Skywork-OR1-Math-7B 5 & ent TJustRL hbx/JustRL-DeepSeek-1.5B 4 & ent TDS7B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 5 & wait
for t in TSky7B:Skywork TJustRL:JustRL TDS7B:R1-7B; do "$E/bin/python" $S/fork_metrics.py --student_ent $F/ent_student_$RT.npz --teacher_ent $F/ent_${t%%:*}_$RT.npz --label "T=0.7 16k rollouts: ds1.5B vs ${t##*:}"; done
echo "L16K_DONE $(date -Is)"
