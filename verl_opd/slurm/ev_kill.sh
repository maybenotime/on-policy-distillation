# kill eval processes whose command line matches $1 (matharena run.py or vllm api_server), excluding this script
for p in $(pgrep -u sya225 -f "$1"); do [ "$p" != "$$" ] && ! grep -q ev_kill /proc/$p/cmdline 2>/dev/null && kill $p && echo "killed $p"; done
