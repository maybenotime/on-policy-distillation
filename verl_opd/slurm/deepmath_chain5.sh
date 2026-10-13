#!/bin/bash
# Phase 3 (login node): once the four DS-student subset runs are done and GPUs 0-2 are idle, and pair-2 labelling is still
# unfinished, stop the single M500 worker at a chunk boundary and relaunch M500 4-way on GPUs 0-3 (resumable via --done_from).
# Peeks every 30 min; stops at both_correct >= 10000 and both_wrong >= 7000; appends CHAIN_ALL_DONE to deepmath_chain4.out.
J=269301; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; KC=10000; KW=7000
D=/home/sya225/opd/verl_data/deepmath/sample_tails.parquet; O=/localscratch/sya225/deepmath; OUT=/home/sya225/opd/verl_data/deepmath
A=Q3_8B; B=M500; BM=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
until grep -q "p1c_R17B ENDED" $SL/driver19b.out 2>/dev/null; do sleep 300; done
grep -q CHAIN_ALL_DONE $SL/deepmath_chain4.out 2>/dev/null && { echo "PHASE3: pair 2 already done, nothing to do"; exit 0; }
until grep -q GEN_END1 $SL/dm1_$A.out 2>/dev/null; do sleep 120; done      # Q3_8B single worker must be finished
echo "==== PHASE3: stopping the single M500 worker $(date -Is)"
tmux kill-session -t dmchain4 2>/dev/null; touch $OUT/STOP_$B
until grep -q GEN_END1 $SL/dm1_$B.out 2>/dev/null; do sleep 120; done; rm -f $OUT/STOP_$B
echo "==== PHASE3: launching 4-way M500 on GPUs 0-3 $(date -Is)"
for i in 0 1 2 3; do tmux new-session -d -s dm${B}_p3_$i "NSHARDS=4 srun --jobid=$J --overlap -c 12 --ntasks=1 bash $SL/deepmath_full.sh $B $BM $i $i qwen3nt > $SL/dm_${B}_p3_$i.out 2>&1"; sleep 20; done
grade_build () {
  srun --jobid=$J --overlap -c 8 --ntasks=1 bash -lc "source /home/sya225/opd/env.sh none; P=\$CODE_EVAL_ENV/bin/python
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_${A}_shard0.jsonl $O/gen_${A}_shard1.jsonl --out $O/grade_$A.jsonl
    \$P $S/deepmath_grade.py --data $D --gen $O/gen_${B}_shard0.jsonl $O/gen_${B}_shard1.jsonl $O/gen_${B}_shard2.jsonl $O/gen_${B}_shard3.jsonl --out $O/grade_$B.jsonl
    \$P $S/deepmath_build.py --data $D --a $O/grade_$A.jsonl --b $O/grade_$B.jsonl --k $KC --name_a $A --name_b $B --outdir $OUT/${A}_$B
    cp $O/grade_$A.jsonl $O/grade_$B.jsonl $OUT/${A}_$B/" 2>&1 | grep -v "module avail\|module load\|teamdynamix" | tee $SL/dm_peek_${A}_$B.txt
  bc=$(grep -o "both correct: [0-9]*" $SL/dm_peek_${A}_$B.txt | grep -o "[0-9]*$"); bw=$(grep -o "both wrong: [0-9]*" $SL/dm_peek_${A}_$B.txt | grep -o "[0-9]*$")
  [ "${bc:-0}" -ge $KC ] && [ "${bw:-0}" -ge $KW ]
}
done_n () { cat $SL/dm_${B}_p3_*.out 2>/dev/null | grep -c GEN_END; }
sleep 1500
while [ "$(done_n)" -lt 4 ]; do
  echo "---- peek $(date -Is)"
  if grade_build; then echo "==== BUCKETS FULL -> STOP $(date -Is)"; touch $OUT/STOP_$B; break; fi
  sleep 1800
done
until [ "$(done_n)" -ge 4 ]; do sleep 120; done
echo "==== PAIR $A + $B GEN DONE $(date -Is)"; grade_build; echo "==== PAIR $A + $B BUILD DONE $(date -Is)"
echo "==== CHAIN_ALL_DONE $(date -Is)" | tee -a $SL/deepmath_chain4.out
