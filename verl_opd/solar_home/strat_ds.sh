source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; R=$SCRATCH/rho
"$E/bin/python" $OPD_HOME/scripts/rho_strat.py \
  --w_stu "$R/ent_stu_p1w_JustRL_sh*.npz" --w_tea "$R/ent_JustRL_p1w_JustRL_sh*.npz" \
  --c_stu "$R/ent_stu_p1c_JustRL_sh*.npz" --c_tea "$R/ent_JustRL_p1c_JustRL_sh*.npz" \
  --label "DS-R1-Distill-1.5B -> JustRL-1.5B"
