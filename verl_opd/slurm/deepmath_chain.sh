#!/bin/bash
# Orchestrator (login node): pair 1 = JustRL + DS7B, pair 2 = Qwen3-8B + Math-Step500. Each model: 2 shards on 2 GPUs.
# Problems are run in "tails" order (hardest-first stream 3:1 with easiest-first stream). Every 20 min the partial
# generations are graded; once both buckets have >= K, STOP files make the generators exit at the next chunk boundary.
J=269301; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; K=10000
D=/home/sya225/opd/verl_data/deepmath/sample_tails.parquet; O=/localscratch/sya225/deepmath; OUT=/home/sya225/opd/verl_data/deepmath
grade_build () {  # A B -> prints build summary; returns 0 if both buckets >= K
  srun --jobid=$J --overlap -c 16 --ntasks=1 bash -lc "source /home/sya225/opd/env.sh none; P=\$CODE_EVAL_ENV/bin/python
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_$1_shard0.jsonl $O/gen_$1_shard1.jsonl --out $O/grade_$1.jsonl
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_$2_shard0.jsonl $O/gen_$2_shard1.jsonl --out $O/grade_$2.jsonl
    \$P $S/deepmath_build.py --data $D --a $O/grade_$1.jsonl --b $O/grade_$2.jsonl --k $K --name_a $1 --name_b $2 --outdir $OUT/$1_$2
    cp $O/grade_$1.jsonl $O/grade_$2.jsonl $OUT/$1_$2/" 2>&1 | grep -v "module avail\|module load\|teamdynamix" | tee $SL/dm_peek_$1_$2.txt
  bc=$(grep -o "both correct: [0-9]*" $SL/dm_peek_$1_$2.txt | grep -o "[0-9]*$"); bw=$(grep -o "both wrong: [0-9]*" $SL/dm_peek_$1_$2.txt | grep -o "[0-9]*$")
  [ "${bc:-0}" -ge $K ] && [ "${bw:-0}" -ge $K ]
}
run_pair () {  # A_NAME A_MODEL A_FAM B_NAME B_MODEL B_FAM
  rm -f $OUT/STOP_$1 $OUT/STOP_$4
  echo "==== PAIR $1 + $4 START $(date -Is)"
  tmux new-session -d -s dm$1_0 "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $1 $2 0 0 $3 > $SL/dm_$1_0.out 2>&1"; sleep 20
  tmux new-session -d -s dm$1_1 "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $1 $2 1 1 $3 > $SL/dm_$1_1.out 2>&1"; sleep 20
  tmux new-session -d -s dm$4_0 "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $4 $5 0 2 $6 > $SL/dm_$4_0.out 2>&1"; sleep 20
  tmux new-session -d -s dm$4_1 "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $4 $5 1 3 $6 > $SL/dm_$4_1.out 2>&1"
  done_n () { cat $SL/dm_$1_0.out $SL/dm_$1_1.out $SL/dm_$4_0.out $SL/dm_$4_1.out 2>/dev/null | grep -c GEN_END; }
  sleep 1200
  while [ "$(done_n)" -lt 4 ]; do
    echo "---- peek $(date -Is)"
    if grade_build $1 $4; then echo "==== BUCKETS FULL for $1 + $4 -> STOP $(date -Is)"; touch $OUT/STOP_$1 $OUT/STOP_$4; break; fi
    sleep 1200
  done
  until [ "$(done_n)" -ge 4 ]; do sleep 120; done
  echo "==== PAIR $1 + $4 GEN DONE $(date -Is)"
  grade_build $1 $4; echo "==== PAIR $1 + $4 BUILD DONE $(date -Is)"
}
run_pair JustRL hbx/JustRL-DeepSeek-1.5B ds DS7B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B ds
run_pair Q3_8B Qwen/Qwen3-8B qwen3nt M500 Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 qwen3nt
echo "==== CHAIN_ALL_DONE $(date -Is)"
