#!/bin/bash
# Reachability probe (DS-1.5B student): each teacher rolls out on the 1000 DAPO pre-probe prompts; the base student
# and the teacher itself score those trajectories.  Waits for the JustRL DeepMath shards to free GPUs 0/1.
source /home/sya225/opd/env.sh none; PY=$CODE_EVAL_ENV/bin/python; S=$OPD_HOME/scripts; F=$OPD_HOME/fork; L=$SCRATCH/logs
export PATH="$CODE_EVAL_ENV/bin:$PATH" TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
D=/home/sya225/opd/verl_data/dapo_math17k/train.parquet; STU=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; O=$SCRATCH/reach; mkdir -p $O
until [ "$(cat /home/sya225/slurm/dm_JustRL_0.out /home/sya225/slurm/dm_JustRL_1.out 2>/dev/null | grep -c GEN_END)" -ge 2 ]; do sleep 120; done
echo "REACH_START $(date -Is)"
roll () { [ -s $F/roll_T$1-dapo8k.jsonl ] || CUDA_VISIBLE_DEVICES=$3 $PY $S/reach_rollout.py --model $2 --family ds --data $D --max_tokens 8192 --out $F/roll_T$1-dapo8k.jsonl > $L/reach_roll_$1.log 2>&1; grep "\[reach\]" $L/reach_roll_$1.log; }
score () { CUDA_VISIBLE_DEVICES=$4 $PY $S/reach_score.py --model $3 --rollouts $1 --out $O/$2.npz --label "$2" 2>&1 | grep "\[score\] " ; }
# GPU 0: JustRL then DS7B rollouts; GPU 1: Skywork rollout — in parallel
( roll JustRL hbx/JustRL-DeepSeek-1.5B 0; roll DS7B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 0 ) &
( roll Sky7B Skywork/Skywork-OR1-Math-7B 1 ) &
wait
echo "ROLLOUTS_DONE $(date -Is)"
# self-baseline: student scores its own pre-probe rollouts
score $F/roll_ds15b-dapo.jsonl student_on_student $STU 0 &
score $F/roll_TJustRL-dapo8k.jsonl student_on_JustRL $STU 1 &
wait
score $F/roll_TDS7B-dapo8k.jsonl student_on_DS7B $STU 0 &
score $F/roll_TSky7B-dapo8k.jsonl student_on_Sky7B $STU 1 &
wait
score $F/roll_TJustRL-dapo8k.jsonl JustRL_on_JustRL hbx/JustRL-DeepSeek-1.5B 0 &
score $F/roll_TDS7B-dapo8k.jsonl DS7B_on_DS7B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 1 &
wait
score $F/roll_TSky7B-dapo8k.jsonl Sky7B_on_Sky7B Skywork/Skywork-OR1-Math-7B 0
echo "REACH_DONE $(date -Is)"
