set -e
cd /home/sya225/opd/scripts
grep -q 'GPU_UTIL' eval_v5.sh || sed -i 's|--tensor-parallel-size "$TP" --dtype bfloat16 --gpu-memory-utilization 0.85|--tensor-parallel-size "$TP" --dtype bfloat16 --gpu-memory-utilization ${GPU_UTIL:-0.85}|' eval_v5.sh
grep -q 'CONCURRENCY' post_run.sh || sed -i 's|-e "s#^max_tokens:.*#max_tokens: ${MAX_TOKENS:-8192}#"|-e "s#^max_tokens:.*#max_tokens: ${MAX_TOKENS:-8192}#" -e "s#^concurrent_requests:.*#concurrent_requests: ${CONCURRENCY:-256}#"|' post_run.sh
grep -n 'GPU_UTIL' eval_v5.sh; grep -n 'CONCURRENCY' post_run.sh
