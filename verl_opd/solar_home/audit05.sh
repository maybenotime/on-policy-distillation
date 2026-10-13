# List, for every FSDP checkpoint on this node, whether a merged HF copy already exists.
set -u
C=/localscratch/sya225/checkpoints
printf "%-52s %10s  %s\n" "FSDP checkpoint" "size" "merged HF copy"
for d in $C/opd_*/global_step_*; do
  [ -d "$d" ] || continue
  run=$(basename $(dirname $d)); st=$(basename $d | sed 's/global_step_//')
  tag=${run#opd_}
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  hf=$C/hf_${tag}_step${st}
  if [ -s "$hf/config.json" ]; then
    printf "%-52s %10s  YES  %s (%s)\n" "$run/step_$st" "$sz" "$(basename $hf)" "$(du -sh $hf 2>/dev/null | cut -f1)"
  else
    printf "%-52s %10s  no\n" "$run/step_$st" "$sz"
  fi
done
echo
echo "=== totals ==="
echo "FSDP checkpoints: $(du -sh $C/opd_* 2>/dev/null | awk '{s+=$1} END {print NR\" dirs\"}')"
du -sh $C 2>/dev/null
df -h /localscratch | tail -1
