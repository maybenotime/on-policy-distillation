#!/bin/bash
# Phase 2 of pair 2: once Qwen3-8B is fully generated and the Light-R1 probe has released GPU 0, restart Math-Step500
# generation 4-way (resumable; STOP at chunk boundaries first), then peek/stop (both_correct >= 10000, both_wrong >= 7000).
J=269301; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; KC=10000; KW=7000
D=/home/sya225/opd/verl_data/deepmath/sample_tails.parquet; O=/localscratch/sya225/deepmath; OUT=/home/sya225/opd/verl_data/deepmath
A=Q3_8B; B=M500; BM=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500; FAM=qwen3nt
until [ "$(cat $SL/dm_${A}_0.out $SL/dm_${A}_1.out 2>/dev/null | grep -c GEN_END)" -ge 2 ]; do sleep 300; done
until grep -q PRE_LIGHT_DONE $SL/pre_dslight.out 2>/dev/null; do sleep 120; done
echo "==== PHASE2: stopping 2-way M500 at chunk boundary $(date -Is)"
touch $OUT/STOP_$B
until [ "$(cat $SL/dm_${B}_0.out $SL/dm_${B}_1.out 2>/dev/null | grep -c GEN_END)" -ge 2 ]; do sleep 120; done
tmux kill-session -t dmchain2 2>/dev/null; tmux kill-session -t stop7k2 2>/dev/null; rm -f $OUT/STOP_$B
echo "==== PHASE2: launching 4-way M500 $(date -Is)"
for i in 0 1 2 3; do
  tmux new-session -d -s dm${B}_4w_$i "NSHARDS=4 srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $B $BM $i $i $FAM > $SL/dm_${B}_4w_$i.out 2>&1"; sleep 20
done
grade_build () {
  srun --jobid=$J --overlap -c 16 --ntasks=1 bash -lc "source /home/sya225/opd/env.sh none; P=\$CODE_EVAL_ENV/bin/python
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_${A}_shard0.jsonl $O/gen_${A}_shard1.jsonl --out $O/grade_$A.jsonl
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_${B}_shard0.jsonl $O/gen_${B}_shard1.jsonl $O/gen_${B}_shard2.jsonl $O/gen_${B}_shard3.jsonl --out $O/grade_$B.jsonl
    \$P $S/deepmath_build.py --data $D --a $O/grade_$A.jsonl --b $O/grade_$B.jsonl --k $KC --name_a $A --name_b $B --outdir $OUT/${A}_$B
    cp $O/grade_$A.jsonl $O/grade_$B.jsonl $OUT/${A}_$B/" 2>&1 | grep -v "module avail\|module load\|teamdynamix" | tee $SL/dm_peek_${A}_$B.txt
  bc=$(grep -o "both correct: [0-9]*" $SL/dm_peek_${A}_$B.txt | grep -o "[0-9]*$"); bw=$(grep -o "both wrong: [0-9]*" $SL/dm_peek_${A}_$B.txt | grep -o "[0-9]*$")
  [ "${bc:-0}" -ge $KC ] && [ "${bw:-0}" -ge $KW ]
}
done_n () { cat $SL/dm_${B}_4w_*.out 2>/dev/null | grep -c GEN_END; }
sleep 1500
while [ "$(done_n)" -lt 4 ]; do
  echo "---- peek $(date -Is)"
  if grade_build; then echo "==== BUCKETS FULL for $A + $B -> STOP $(date -Is)"; touch $OUT/STOP_$B; break; fi
  sleep 1500
done
until [ "$(done_n)" -ge 4 ]; do sleep 120; done
echo "==== PAIR $A + $B GEN DONE $(date -Is)"
grade_build; echo "==== PAIR $A + $B BUILD DONE $(date -Is)"; echo "==== CHAIN_ALL_DONE $(date -Is)"
