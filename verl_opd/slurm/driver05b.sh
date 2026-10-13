#!/bin/bash
# login-node driver, cs-venus-05 (job 269604): the two weak-teacher DS runs (R1-Distill-7B on p1w / p1c), after the
# three new-teacher runs and their evals free the GPUs.  p1w on GPUs 3,4,5; p1c on GPUs 0,1,2; evals on the same sets.
J=269604; SL=/home/sya225/slurm; S=/home/sya225/opd/scripts; D=/home/sya225/opd/verl_data
DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; T7=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
echo "DRIVER05B_START $(date -Is)"
n_done () { cat $SL/$1.out 2>/dev/null | grep -c EVAL_DONE; }
# GPUs 3,4,5 free once the Nemo/Light evals are done (ev_g3 has 2 evals, ev_g4 and ev_g5 one each)
until grep -q LIGHT_ENDED $SL/driver05.out 2>/dev/null && [ "$(n_done ev_g3)" -ge 2 ] && [ "$(n_done ev_g4)" -ge 1 ] && [ "$(n_done ev_g5)" -ge 1 ]; do sleep 600; done
echo "GPUS_345_FREE $(date -Is)"; sleep 60
tmux new-session -d -s tr_p1w_R17B "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain_dm.sh ds15b_fromDS7B_dmP1W $DS $T7 $D/deepmath_p1w 3,4,5 /tmp/ray_p1w > $SL/tr_p1w_R17B.out 2>&1"
# GPUs 0,1,2 free once DeepScaleR has ended and its evals (ev_g0, ev_g1) are done
until grep -q DSCALER_ENDED $SL/driver05.out 2>/dev/null && [ "$(n_done ev_g0)" -ge 1 ] && [ "$(n_done ev_g1)" -ge 1 ]; do sleep 600; done
echo "GPUS_012_FREE $(date -Is)"; sleep 90
tmux new-session -d -s tr_p1c_R17B "srun --jobid=$J --overlap -c 30 --ntasks=1 bash $S/chain_dm.sh ds15b_fromDS7B_dmP1C $DS $T7 $D/deepmath_p1c 0,1,2 /tmp/ray_p1c > $SL/tr_p1c_R17B.out 2>&1"
until grep -q '######## END' $SL/tr_p1w_R17B.out 2>/dev/null; do sleep 300; done; echo "p1w_R17B ENDED $(date -Is)"; sleep 60
tmux new-session -d -s ev_p1w_R17B "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05.sh ds15b_fromDS7B_dmP1W 200 3 8111 p1w_R17B; bash $SL/eval05.sh ds15b_fromDS7B_dmP1W 50 4 8112 p1w_R17B' > $SL/ev_p1w_R17B.out 2>&1"
until grep -q '######## END' $SL/tr_p1c_R17B.out 2>/dev/null; do sleep 300; done; echo "p1c_R17B ENDED $(date -Is)"; sleep 60
tmux new-session -d -s ev_p1c_R17B "srun --jobid=$J --overlap -c 8 --ntasks=1 bash -c 'bash $SL/eval05.sh ds15b_fromDS7B_dmP1C 200 0 8113 p1c_R17B; bash $SL/eval05.sh ds15b_fromDS7B_dmP1C 50 1 8114 p1c_R17B' > $SL/ev_p1c_R17B.out 2>&1"
echo "DRIVER05B_ALL_LAUNCHED $(date -Is)"
