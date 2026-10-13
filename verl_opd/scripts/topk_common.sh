# shared by topk19.sh / topk05.sh
source /home/sya225/opd/env.sh none
E=$CODE_EVAL_ENV; F=$OPD_HOME/fork; D=$SCRATCH/fork_topk; L=$SCRATCH/logs; mkdir -p $D
export PATH="$E/bin:$PATH" TOKENIZERS_PARALLELISM=false
dump () {  # gpu roll_tag model out_name
  [ -s "$D/$4.npz" ] && { echo "[$4] exists"; return; }
  echo "######## [$4] gpu=$1 model=$3 start $(date -Is) ########"
  CUDA_VISIBLE_DEVICES=$1 "$E/bin/python" $OPD_HOME/scripts/fork_topk.py --model "$3" \
      --rollouts "$F/roll_$2.jsonl" --out "$D/$4.npz" --k 32 > $L/topk_$4.log 2>&1
  echo "######## [$4] rc=$? $(date -Is) ########"
}
met () { "$E/bin/python" $OPD_HOME/scripts/overlap_metrics.py --student "$D/$1.npz" --teacher "$D/$2.npz" --label "$3"; }
