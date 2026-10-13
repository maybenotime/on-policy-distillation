set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs
TAG=ds15b_tpStu_uniform; BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
CK=$SCRATCH/checkpoints/opd_$TAG/global_step_50/actor; HF=$SCRATCH/checkpoints/hf_${TAG}_step50
[ -d $CK ] || { echo "NO CHECKPOINT $CK"; exit 1; }
if [ ! -s $HF/config.json ]; then
  CUDA_VISIBLE_DEVICES=3 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
echo "MERGE_DONE uniform s50 $(date -Is)"
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
bash $S/teacher_eval.sh dsTP_Stu_uniform-s50-16k $HF 3 8110 $BASE > $L/teval_uni50_early.log 2>&1
echo "EVAL_DONE uniform-s50-early rc=$? $(date -Is)"
