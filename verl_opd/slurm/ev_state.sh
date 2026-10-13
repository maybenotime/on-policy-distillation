# state of eval processes on this node (safe: patterns live in this file, not in the srun command line)
pgrep -u sya225 -af "api_server|scripts/run.py" | grep -v pgrep | grep -oE "local/[a-zA-Z0-9_.]+|--model [^ ]+" | sort | uniq -c
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr "\n" " "; echo
