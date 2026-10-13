for L in train_math17b_fromK4Bmath1200_16k train_17b_fromK4Bmath1200_16k; do
  f=/localscratch/sya225/logs/$L.log
  n=$(grep -an "python3 -m verl.trainer.main_ppo" $f | tail -1 | cut -d: -f1)
  tail -n +$n $f > /tmp/chk_$L.txt
  echo "$L ray=$(grep -ac 'Started a local Ray' /tmp/chk_$L.txt) loaded=$(grep -a 'Loaded model from' /tmp/chk_$L.txt | tail -1 | grep -oE 'global_step_[0-9]+') step=$(grep -aoE ' step:[0-9]+' /tmp/chk_$L.txt | tail -1) err=$(grep -aEc 'Traceback|timed out' /tmp/chk_$L.txt)"
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
