pkill -f run_opd_code.sh; pkill -f main_ppo; pkill -f 'ray::'; pkill -f vLLMHttpServer; pkill -f raylet; pkill -f gcs_server; sleep 15
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
rm -rf /localscratch/sya225/checkpoints/opd_math_17b_fromK4Bmath1200 /tmp/ray_math_K4Bmath1200
mv /localscratch/sya225/logs/train_math17b_fromK4Bmath1200.log /localscratch/sya225/logs/train_math17b_fromK4Bmath1200.restart.log
