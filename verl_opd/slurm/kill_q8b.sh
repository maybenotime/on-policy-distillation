# kill only the 8B training run's processes (main_ppo + its ray/vLLM children), not the eval api_server on GPU 0
for p in $(pgrep -u sya225 -f "main_ppo"); do kill $p; done; sleep 15
for p in $(pgrep -u sya225 -f "ray::|vLLMHttpServer|raylet|gcs_server|ray_q8b"); do kill -9 $p 2>/dev/null; done; sleep 5
pgrep -u sya225 -af "main_ppo|vLLMHttpServer|raylet" | grep -v pgrep | wc -l
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo
