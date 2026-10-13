# print per-step probe metrics of the DeepSeek run (every Nth step), plus timing/clip and ckpts
L=/localscratch/sya225/logs/train_ds15b_fromJustRL_dapo_16k.log; N=${1:-10}
date +%H:%M; grep -ac Traceback $L
grep -a " step:[0-9]* " $L | awk -v n=$N 'NR==1 || NR%n==0 {print}' | while read -r line; do
  s=$(echo "$line" | grep -oE "step:[0-9]+" | head -1)
  v=$(echo "$line" | tr " " "\n" | grep -aE "probe/(overlap16|rho|fork_overlap|dH|student_entropy|teacher_entropy_topk)|distillation/abs_loss|response_length/(mean|clip_ratio)|timing_s/step" | sed -E "s/actor\///; s/np.float64\(([-0-9.e]+)\)/\1/" | awk -F: '{printf "%s=%.3f ", $1, $2}')
  echo "$s $v"
done
ls /localscratch/sya225/checkpoints/opd_ds15b_fromJustRL_dapo_16k/ 2>/dev/null | tr "\n" " "; echo
