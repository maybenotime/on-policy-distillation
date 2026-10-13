source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; R=$SCRATCH/rho
"$E/bin/python" $OPD_HOME/scripts/k1_pos_agg.py \
  --w "$R/k1_p1w_JustRL_sh*.npz" --c "$R/k1_p1c_JustRL_sh*.npz" \
  --bins 0,128,512,1024,2048 --label "DS-1.5B vs JustRL  (both-wrong vs both-correct)"
