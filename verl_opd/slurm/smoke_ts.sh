set -u
source /home/sya225/opd/env.sh none
rm -rf $SCRATCH/checkpoints/opd_q3i17_ts_smoke $SCRATCH/logs/train_q3i17_ts_smoke.log
SMOKE=1 TOKEN_SELECT=mismatch TOKEN_FRAC=0.3 bash /home/sya225/opd/scripts/chain19_ts.sh q3i17_ts_smoke Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500 0,1,2 /tmp/ray_tssmoke
echo "SMOKE_TS_END $(date -Is)"
L=$SCRATCH/logs/train_q3i17_ts_smoke.log
echo "--- selection metrics seen:"; grep -o "select/[a-z_0-9]*:[^ ]*" $L | tail -8
echo "--- steps logged: $(grep -c 'training/global_step' $L)"
grep -aE "Traceback|Error" $L | tail -3
