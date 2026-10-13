# usage: bash chk_run.sh <train log basename>  -> one line: tb=<n> step=<s> key=val ...
L=/localscratch/sya225/logs/$1.log
printf "tb=%s " "$(grep -ac Traceback $L)"
grep -a " step:[0-9]* - " $L | tail -n 1 | tr " " "\n" | grep -aE "^(step:|actor/probe/(student_entropy|rho|overlap16|dH|sampled_in_topk)|actor/grad_norm|actor/distillation/abs_loss|response_length/(mean|clip_ratio)|timing_s/step)" | sed -E 's/np.float64\(([-0-9.e]+)\)/\1/; s#actor/##; s#probe/##; s#distillation/##; s#response_length/#rl_#' | awk -F: '{printf "%s=%s ", $1, $2}'
echo
