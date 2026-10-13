source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; R=$SCRATCH/rho
"$E/bin/python" $OPD_HOME/scripts/rho_strat.py \
  --w_stu "$R/ent_stu_p2w_M500_sh*.npz" --w_tea "$R/ent_M500_p2w_M500_sh*.npz" \
  --c_stu "$R/ent_stu_p2c_M500_sh*.npz" --c_tea "$R/ent_M500_p2c_M500_sh*.npz" \
  --label "Qwen3-1.7B -> Math-Step500"
