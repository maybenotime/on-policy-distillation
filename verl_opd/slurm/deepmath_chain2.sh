#!/bin/bash
# Pair 2 only: Qwen3-8B + Math-Step500 (non-thinking).  done_n bug fixed (names passed explicitly).
J=269301; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; K=10000
D=/home/sya225/opd/verl_data/deepmath/sample_tails.parquet; O=/localscratch/sya225/deepmath; OUT=/home/sya225/opd/verl_data/deepmath
A=Q3_8B; AM=Qwen/Qwen3-8B; B=M500; BM=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500; FAM=qwen3nt
grade_build () {
  srun --jobid=$J --overlap -c 16 --ntasks=1 bash -lc "source /home/sya225/opd/env.sh none; P=\$CODE_EVAL_ENV/bin/python
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_${A}_shard0.jsonl $O/gen_${A}_shard1.jsonl --out $O/grade_$A.jsonl
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_${B}_shard0.jsonl $O/gen_${B}_shard1.jsonl --out $O/grade_$B.jsonl
    \$P $S/deepmath_build.py --data $D --a $O/grade_$A.jsonl --b $O/grade_$B.jsonl --k $K --name_a $A --name_b $B --outdir $OUT/${A}_$B
    cp $O/grade_$A.jsonl $O/grade_$B.jsonl $OUT/${A}_$B/" 2>&1 | grep -v "module avail\|module load\|teamdynamix" | tee $SL/dm_peek_${A}_$B.txt
  bc=$(grep -o "both correct: [0-9]*" $SL/dm_peek_${A}_$B.txt | grep -o "[0-9]*$"); bw=$(grep -o "both wrong: [0-9]*" $SL/dm_peek_${A}_$B.txt | grep -o "[0-9]*$")
  [ "${bc:-0}" -ge $K ] && [ "${bw:-0}" -ge $K ]
}
done_n () { cat $SL/dm_${A}_0.out $SL/dm_${A}_1.out $SL/dm_${B}_0.out $SL/dm_${B}_1.out 2>/dev/null | grep -c GEN_END; }
rm -f $OUT/STOP_$A $OUT/STOP_$B
echo "==== PAIR $A + $B START $(date -Is)"
tmux new-session -d -s dm${A}_0 "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $A $AM 0 0 $FAM > $SL/dm_${A}_0.out 2>&1"; sleep 20
tmux new-session -d -s dm${A}_1 "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $A $AM 1 1 $FAM > $SL/dm_${A}_1.out 2>&1"; sleep 20
tmux new-session -d -s dm${B}_0 "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $B $BM 0 2 $FAM > $SL/dm_${B}_0.out 2>&1"; sleep 20
tmux new-session -d -s dm${B}_1 "srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $B $BM 1 3 $FAM > $SL/dm_${B}_1.out 2>&1"
sleep 900
while [ "$(done_n)" -lt 4 ]; do
  echo "---- peek $(date -Is)"
  if grade_build; then echo "==== BUCKETS FULL for $A + $B -> STOP $(date -Is)"; touch $OUT/STOP_$A $OUT/STOP_$B; break; fi
  sleep 1200
done
until [ "$(done_n)" -ge 4 ]; do sleep 120; done
echo "==== PAIR $A + $B GEN DONE $(date -Is)"
grade_build; echo "==== PAIR $A + $B BUILD DONE $(date -Is)"
echo "==== CHAIN_ALL_DONE $(date -Is)"
