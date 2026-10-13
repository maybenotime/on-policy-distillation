# Delete only FSDP checkpoints that already have a merged HF copy, and never the one a run
# would resume from (latest_checkpointed_iteration.txt).
set -u
C=/localscratch/sya225/checkpoints
TODEL=""
for d in $C/opd_*/global_step_*; do
  [ -d "$d" ] || continue
  run=$(basename $(dirname $d)); st=$(basename $d | sed 's/global_step_//')
  tag=${run#opd_}
  [ -s "$C/hf_${tag}_step${st}/config.json" ] || continue      # no merged copy -> keep
  latest=$(cat "$C/$run/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
  if [ "$st" = "$latest" ]; then
    echo "KEEP (resume point) $run/global_step_$st"
    continue
  fi
  echo "DELETE $run/global_step_$st  ($(du -sh $d | cut -f1))  [merged: hf_${tag}_step${st}]"
  TODEL="$TODEL $d"
done
echo
echo "before: $(df -h /localscratch | tail -1 | awk '{print $4\" free\"}')"
for d in $TODEL; do rm -rf "$d"; done
echo "after:  $(df -h /localscratch | tail -1 | awk '{print $4\" free\"}')"
du -sh $C
