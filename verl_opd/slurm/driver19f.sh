#!/bin/bash
# cs-venus-19 (job 270249): entropy-shape transplant, main pair (student Qwen3-1.7B; A = Qwen3-4B, B = Math-Step500).
# Waits for smoke test 3 (bf16 teachers) to pass, then runs on GPUs 0,1,2 sequentially:
#   run3  tpA4B_shapeM500   A ⊕ shape(B)
#   run5  tpA4B_uniform     A / T*  (T* from slurm/tp_uniform_T.txt: mean H_A(T*) = mean H_B)
#   run4  tpM500_shapeA4B   B ⊕ shape(A)
#   run6  tpA4B_shapeStu    A ⊕ shape(student)
# Each 200 steps; s200 8k eval on GPU 3 right after each run (eval19.sh, Qwen3-1.7B base).
J=270249; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts
A=Qwen/Qwen3-4B; B=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500; Q3=Qwen/Qwen3-1.7B
ended () { grep -q '######## END' $SL/$1.out 2>/dev/null; }
run () { local NAME=$1 MODE=$2 LA=$3 LB=$4 T=$5
  echo "$NAME START $(date -Is)"
  tmux new-session -d -s tr_$NAME "srun --jobid=$J --overlap -c 40 --ntasks=1 bash $S/chain19_tp.sh q3i17_$NAME $MODE $LA $LB 0,1,2 /tmp/ray_$NAME $T 200 > $SL/tr_$NAME.out 2>&1"
  until ended tr_$NAME; do sleep 300; done; echo "$NAME ENDED $(date -Is)"; sleep 60
  tmux new-session -d -s ev_$NAME "srun --jobid=$J --overlap -c 8 --ntasks=1 bash $SL/eval19.sh q3i17_$NAME 200 3 $6 $NAME $Q3 8192 > $SL/ev_$NAME.out 2>&1"
  echo "$NAME EVAL launched on gpu 3 $(date -Is)"; sleep 60; }
echo "DRIVER19F_START $(date -Is)"
until grep -q "END smoke_tp rc=0" $SL/smoke_tp3.out 2>/dev/null; do sleep 60; done
# the "Flash Attention 2 only supports fp16/bf16" warning is emitted at construction time (fp32 default dtype) and is spurious; the tp_core print is the real check
if ! srun --jobid=$J --overlap -c 1 --ntasks=1 grep -aq "param dtype torch.bfloat16, attn flash_attention_2" /localscratch/sya225/logs/train_smoke_tp.log 2>/dev/null; then echo "SMOKE3 teachers not bf16/flash -> abort"; exit 1; fi
echo "SMOKE3_OK $(date -Is)"
until [ -s $SL/tp_uniform_T.txt ]; do sleep 60; done; TU=$(cat $SL/tp_uniform_T.txt)
run tpA4B_shapeM500 shape $A $B 1.0 8160
run tpA4B_uniform   uniform $A - $TU 8162
run tpM500_shapeA4B shape $B $A 1.0 8164
run tpA4B_shapeStu  student $A - 1.0 8166
echo "DRIVER19F_ALL_TRAINED $(date -Is)"
