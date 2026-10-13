# kill the older duplicate matharena process for Q3_1.7B_Base_16k (stale evpar branch); show s50 progress
pids=$(pgrep -u sya225 -f "local/Q3_1.7B_Base_16k" | sort -n); n=$(echo "$pids" | wc -w)
if [ "$n" -gt 1 ]; then old=$(echo "$pids" | head -1); echo "killing stale matharena pid $old"; kill $old; fi
sleep 3; pgrep -u sya225 -af "scripts/run.py" | grep -v pgrep | grep -oE "local/[a-zA-Z0-9_.]+" | sort | uniq -c
for t in q3b17-BaseGRPO-k3-s50-16k Q3-1.7B-Base-16k; do f=/localscratch/sya225/logs/v5_$t.log; echo "$t: $(grep -aoE 'Running: [0-9]+ reqs, Waiting: [0-9]+ reqs, GPU KV cache usage: [0-9.]+%' $f | tail -n 1) | gen tok/s: $(grep -aoE 'Avg generation throughput: [0-9.]+' $f | tail -n 1)"; done
