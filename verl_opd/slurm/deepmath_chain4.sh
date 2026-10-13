#!/bin/bash
# GPU-3-only continuation of DeepMath pair 2 (Qwen3-8B + Math-Step500): finish Q3_8B, then M500, peeking every 30 min;
# stop when both_correct >= 10000 and both_wrong >= 7000.  Waits for the 4-way shards to exit first.
J=269301; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; KC=10000; KW=7000
D=/home/sya225/opd/verl_data/deepmath/sample_tails.parquet; O=/localscratch/sya225/deepmath; OUT=/home/sya225/opd/verl_data/deepmath
A=Q3_8B; AM=Qwen/Qwen3-8B; B=M500; BM=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
until [ "$(cat $SL/dm_${A}_0.out $SL/dm_${A}_1.out $SL/dm_${B}_0.out $SL/dm_${B}_1.out 2>/dev/null | grep -c GEN_END)" -ge 4 ]; do sleep 120; done
rm -f $OUT/STOP_$A $OUT/STOP_$B; echo "==== CHAIN4: all 2-way shards exited $(date -Is)"
grade_build () {
  srun --jobid=$J --overlap -c 8 --ntasks=1 bash -lc "source /home/sya225/opd/env.sh none; P=\$CODE_EVAL_ENV/bin/python
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_${A}_shard0.jsonl $O/gen_${A}_shard1.jsonl --out $O/grade_$A.jsonl
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_${B}_shard0.jsonl $O/gen_${B}_shard1.jsonl --out $O/grade_$B.jsonl
    \$P $S/deepmath_build.py --data $D --a $O/grade_$A.jsonl --b $O/grade_$B.jsonl --k $KC --name_a $A --name_b $B --outdir $OUT/${A}_$B
    cp $O/grade_$A.jsonl $O/grade_$B.jsonl $OUT/${A}_$B/" 2>&1 | grep -v "module avail\|module load\|teamdynamix" | tee $SL/dm_peek_${A}_$B.txt
  bc=$(grep -o "both correct: [0-9]*" $SL/dm_peek_${A}_$B.txt | grep -o "[0-9]*$"); bw=$(grep -o "both wrong: [0-9]*" $SL/dm_peek_${A}_$B.txt | grep -o "[0-9]*$")
  [ "${bc:-0}" -ge $KC ] && [ "${bw:-0}" -ge $KW ]
}
# 1) finish Qwen3-8B over all rows (fast) on GPU 3
tmux new-session -d -s dm1_$A "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full1.sh $A $AM 3 qwen3nt > $SL/dm1_$A.out 2>&1"
until grep -q GEN_END1 $SL/dm1_$A.out 2>/dev/null; do sleep 300; done; echo "==== CHAIN4: $A complete $(date -Is)"
# 2) M500 on GPU 3 with peeks
tmux new-session -d -s dm1_$B "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full1.sh $B $BM 3 qwen3nt > $SL/dm1_$B.out 2>&1"
sleep 1800
while ! grep -q GEN_END1 $SL/dm1_$B.out 2>/dev/null; do
  echo "---- peek $(date -Is)"
  if grade_build; then echo "==== BUCKETS FULL -> STOP $(date -Is)"; touch $OUT/STOP_$B; break; fi
  sleep 1800
done
until grep -q GEN_END1 $SL/dm1_$B.out 2>/dev/null; do sleep 120; done
echo "==== PAIR $A + $B GEN DONE $(date -Is)"; grade_build; echo "==== PAIR $A + $B BUILD DONE $(date -Is)"; echo "==== CHAIN_ALL_DONE $(date -Is)"
