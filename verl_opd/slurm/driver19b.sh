#!/bin/bash
# cs-venus-19 (job 269301): all eight DeepMath-subset OPD runs, sequential on GPUs 0,1,2 (GPU 3 = pair-2 generation, then evals).
# Order: p1 four (data ready) -> p2 four (after pair-2 labelling ends).  Evals are queued and run on GPU 3 after CHAIN_ALL_DONE.
J=269301; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; D=/home/sya225/opd/verl_data; Q=$SL/eval_queue19.txt; : > $Q
echo "DRIVER19B_START $(date -Is)"
until [ "$(cat $SL/dm_Q3_8B_0.out $SL/dm_Q3_8B_1.out $SL/dm_M500_0.out 2>/dev/null | grep -c GEN_END)" -ge 3 ]; do sleep 120; done
echo "GPUS_012_FREE $(date -Is)"; sleep 60
# eval queue runner (GPU 3): starts consuming once the generation has finished
tmux new-session -d -s evq19 "bash $SL/eval_queue19.sh > $SL/eval_queue19.out 2>&1"
run () {  # TAG STUDENT TEACHER DATA NAME BASE MAXTOK
  tmux new-session -d -s tr_$5 "srun --jobid=$J --overlap -c 40 --ntasks=1 bash $S/chain_dm.sh $1 $2 $3 $4 0,1,2 /tmp/ray_dm > $SL/tr_$5.out 2>&1"
  until grep -q '######## END' $SL/tr_$5.out 2>/dev/null; do sleep 300; done; echo "$5 ENDED $(date -Is)"; sleep 60
  echo "$1 $5 $6 $7" >> $Q
}
DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; T7=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B; Q3=Qwen/Qwen3-1.7B; M500=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
run ds15b_fromJustRL_dmP1W $DS hbx/JustRL-DeepSeek-1.5B $D/deepmath_p1w p1w_JustRL $DS 16384
run ds15b_fromJustRL_dmP1C $DS hbx/JustRL-DeepSeek-1.5B $D/deepmath_p1c p1c_JustRL $DS 16384
run ds15b_fromDS7B_dmP1W   $DS $T7 $D/deepmath_p1w p1w_R17B $DS 16384
run ds15b_fromDS7B_dmP1C   $DS $T7 $D/deepmath_p1c p1c_R17B $DS 16384
until grep -q CHAIN_ALL_DONE $SL/deepmath_chain4.out 2>/dev/null; do sleep 600; done
echo "PAIR2_DATA_READY $(date -Is)"
srun --jobid=$J --overlap -c 2 --ntasks=1 bash -lc "source /home/sya225/opd/env.sh none; P=\$CODE_EVAL_ENV/bin/python; B=$D/deepmath/Q3_8B_M500
  \$P $S/deepmath_to_verl.py --src \$(ls \$B/both_wrong_*.jsonl | sort -t_ -k3 -n | tail -1) --n 7000 --out $D/deepmath_p2w
  \$P $S/deepmath_to_verl.py --src \$(ls \$B/both_correct_*.jsonl | sort -t_ -k3 -n | tail -1) --n 7000 --out $D/deepmath_p2c" 2>&1 | grep wrote
run q3i17_fromM500_dmP2W $Q3 $M500 $D/deepmath_p2w p2w_M500 $Q3 8192
run q3i17_fromM500_dmP2C $Q3 $M500 $D/deepmath_p2c p2c_M500 $Q3 8192
run q3i17_from8B_dmP2W   $Q3 Qwen/Qwen3-8B $D/deepmath_p2w p2w_Q8B $Q3 8192
run q3i17_from8B_dmP2C   $Q3 Qwen/Qwen3-8B $D/deepmath_p2c p2c_Q8B $Q3 8192
echo "DRIVER19B_ALL_TRAINED $(date -Is)"
