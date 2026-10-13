set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
HF=$SCRATCH/checkpoints/hf_ds15b_tpStu_uniform_step200
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025"
bash $S/teacher_eval.sh dsTP_Stu_uniform-s200c-16k $HF 2 8112 $BASE > $L/teval_uni200c_retry.log 2>&1
echo "EVAL_DONE s200c-retry rc=$? $(date -Is)"
