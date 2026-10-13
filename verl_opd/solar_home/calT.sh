set -u
source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; S=$OPD_HOME/scripts; F=$OPD_HOME/fork
export PATH="$E/bin:$PATH" CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false
# tp_uniformT.py expects the reference entropies keyed 'ent<i>' (fork_topk layout);
# fork_entropy.py writes them keyed '<i>', so rekey into a scratch copy first.
EB=$SCRATCH/rho/entb_JustRL_ds15b-dapo.npz
[ -s $EB ] || "$E/bin/python" - <<'PY'
import numpy as np, os
src = "/home/sya225/opd/fork/ent_TJustRL_ds15b-dapo.npz"
z = np.load(src)
out = {f"ent{k}": z[k] for k in z.files}
dst = os.environ["SCRATCH"] + "/rho/entb_JustRL_ds15b-dapo.npz"
np.savez_compressed(dst, **out)
print("rekeyed", len(out), "->", dst)
PY
"$E/bin/python" $S/tp_uniformT_sdpa.py --model_a deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --rollouts $F/roll_ds15b-dapo.jsonl --ent_b $EB --n 300 \
  --out /home/sya225/slurm/tp_uniform_T_ds_stu.txt 2>&1 | tail -6
echo "CALT_DONE $(date -Is)"
