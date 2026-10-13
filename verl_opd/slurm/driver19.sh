#!/bin/bash
# login-node driver, cs-venus-19 (job 269301): six DeepMath-subset OPD runs, sequential on GPUs 0,1,2; evals (s200, s50) on GPU 3.
J=269301; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; D=/home/sya225/opd/verl_data
echo "DRIVER19_START $(date -Is)"
until grep -q CHAIN_ALL_DONE $SL/deepmath_chain3.out 2>/dev/null; do sleep 600; done
echo "PAIR2_DATA_READY $(date -Is)"
srun --jobid=$J --overlap -c 2 --ntasks=1 bash -lc "source /home/sya225/opd/env.sh none; P=\$CODE_EVAL_ENV/bin/python; B=$D/deepmath/Q3_8B_M500
  \$P $S/deepmath_to_verl.py --src \$(ls \$B/both_wrong_*.jsonl | sort -t_ -k3 -n | tail -1) --n 7000 --out $D/deepmath_p2w
  \$P $S/deepmath_to_verl.py --src \$(ls \$B/both_correct_*.jsonl | sort -t_ -k3 -n | tail -1) --n 7000 --out $D/deepmath_p2c" 2>&1 | grep wrote
PORT=8100
run_and_eval () {  # TAG STUDENT TEACHER DATA NAME BASE MAXTOK
  tmux new-session -d -s tr_$5 "srun --jobid=$J --overlap -c 40 --ntasks=1 bash $S/chain_dm.sh $1 $2 $3 $4 0,1,2 /tmp/ray_dm > $SL/tr_$5.out 2>&1"
  until grep -q '######## END' $SL/tr_$5.out 2>/dev/null; do sleep 300; done; echo "$5 ENDED $(date -Is)"; sleep 60
  PORT=$((PORT+2))
  tmux new-session -d -s ev_$5 "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval19.sh $1 200 3 $PORT $5 $6 $7; bash $SL/eval19.sh $1 50 3 $((PORT+1)) $5 $6 $7' > $SL/ev_$5.out 2>&1"
}
DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; Q3=Qwen/Qwen3-1.7B; M500=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
run_and_eval ds15b_fromJustRL_dmP1W $DS hbx/JustRL-DeepSeek-1.5B $D/deepmath_p1w p1w_JustRL $DS 16384
run_and_eval ds15b_fromJustRL_dmP1C $DS hbx/JustRL-DeepSeek-1.5B $D/deepmath_p1c p1c_JustRL $DS 16384
run_and_eval q3i17_fromM500_dmP2W $Q3 $M500 $D/deepmath_p2w p2w_M500 $Q3 8192
run_and_eval q3i17_fromM500_dmP2C $Q3 $M500 $D/deepmath_p2c p2c_M500 $Q3 8192
run_and_eval q3i17_from8B_dmP2W $Q3 Qwen/Qwen3-8B $D/deepmath_p2w p2w_Q8B $Q3 8192
run_and_eval q3i17_from8B_dmP2C $Q3 Qwen/Qwen3-8B $D/deepmath_p2c p2c_Q8B $Q3 8192
echo "DRIVER19_ALL_LAUNCHED $(date -Is)"
