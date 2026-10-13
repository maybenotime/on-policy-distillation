set -u
source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; S=$OPD_HOME/scripts; R=$SCRATCH/rho; mkdir -p $R
export PATH="$E/bin:$PATH" CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
D=/home/sya225/opd/verl_data/deepmath/verl_format/p1w/train.parquet
"$E/bin/python" -c "import pandas as pd; d=pd.read_parquet('$D'); print('rows',len(d))"
"$E/bin/python" $S/rho_rollout.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --family ds \
   --data $D --offset 0 --n 24 --seed 0 --max_tokens 2048 --out $R/smoke.jsonl 2>&1 | tail -3
"$E/bin/python" $S/fork_entropy.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --rollouts $R/smoke.jsonl --out $R/ent_stu_smoke_sh0.npz --chunk 256 2>&1 | tail -2
"$E/bin/python" $S/fork_entropy.py --model hbx/JustRL-DeepSeek-1.5B --rollouts $R/smoke.jsonl --out $R/ent_JustRL_smoke_sh0.npz --chunk 256 2>&1 | tail -2
"$E/bin/python" $S/rho_agg.py --student "$R/ent_stu_smoke_sh*.npz" --teacher "$R/ent_JustRL_smoke_sh*.npz" --label "SMOKE p1w vs JustRL"
echo SMOKE_RC=$?
