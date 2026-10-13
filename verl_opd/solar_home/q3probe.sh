F=/localscratch/sya225/logs/train_q3i17_tpStu_shapeM500.log
for s in 1 5 10 15 20 25 30; do
  L=$(grep -m1 "training/global_step:$s$" "$F" 2>/dev/null)
  [ -z "$L" ] && continue
  v() { echo "$L" | grep -oE "$1:(np\.float64\()?[0-9.e+-]*" | head -1 | sed -E 's/.*[:(]//'; }
  echo "s$s H_A=$(v 'actor/tp/H_A') H_tgt=$(v 'actor/tp/H_target') H_til=$(v 'actor/tp/H_tilde') T=$(v 'actor/tp/T_mean') rho_sB=$(v 'actor/tp/rho_s_B') ov_sB=$(v 'actor/tp/overlap16_sB') k1=$(v 'actor/tp/abs_k1_A') Hstu=$(v 'actor/entropy') inTopK=$(v 'actor/probe/sampled_in_topk') len=$(v 'response_length/mean')"
done
