#!/bin/bash
# 264439 GPU 2 (shared): length probe of the Math-1200 teacher and of the math-run student at step 100
set -u; source /home/USER/opd/env.sh none; E=$CODE_EVAL_ENV; export PATH="$E/bin:$PATH" VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=2
S=$OPD_HOME/scripts; O=$OPD_HOME/fork/probe; mkdir -p $O
"$E/bin/python" $S/len_probe.py --model Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200 --n 48 --max_tokens 16384 --util 0.3 --out $O/T_K4Bmath1200.jsonl 2>&1 | grep -a '\[probe\]'
CK=$SCRATCH/checkpoints/opd_math_17b_fromK4Bmath1200/global_step_100/actor; HF=$SCRATCH/checkpoints/hf_opd_1.7b_math_fromK4Bmath1200_step100
[ -s $HF/config.json ] || { "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1; "$E/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('Qwen/Qwen3-1.7B').save_pretrained('$HF')"; }
"$E/bin/python" $S/len_probe.py --model $HF --n 48 --max_tokens 16384 --util 0.3 --out $O/S_math1200_step100.jsonl 2>&1 | grep -a '\[probe\]'
"$E/bin/python" $S/len_probe.py --model Qwen/Qwen3-1.7B --n 48 --max_tokens 16384 --util 0.3 --out $O/S_base.jsonl 2>&1 | grep -a '\[probe\]'
echo "PROBE_DONE $(date -Is)"
