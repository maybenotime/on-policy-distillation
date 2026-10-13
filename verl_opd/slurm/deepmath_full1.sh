#!/bin/bash
# bash deepmath_full1.sh <NAME> <MODEL> <GPU> <FAMILY>  — single worker over ALL rows, resuming from every gen_<NAME>_shard*.jsonl
source /home/sya225/opd/env.sh none; PY=$CODE_EVAL_ENV/bin/python; S=$OPD_HOME/scripts; export PATH="$CODE_EVAL_ENV/bin:$PATH" TOKENIZERS_PARALLELISM=false
D=/home/sya225/opd/verl_data/deepmath/sample_tails.parquet; O=$SCRATCH/deepmath; N=$1; M=$2; G=$3; FAM=${4:-ds}
echo "START1 $N gpu $G family $FAM $(date -Is)"
CUDA_VISIBLE_DEVICES=$G $PY $S/deepmath_gen.py --model $M --data $D --shard 0 --nshards 1 --chunk 1000 --max_tokens 16384 --family $FAM \
    --done_from "$O/gen_${N}_shard*.jsonl" --stop_file /home/sya225/opd/verl_data/deepmath/STOP_$N --out $O/gen_${N}_shard0.jsonl > $SCRATCH/logs/deepmath_gen1_${N}.log 2>&1
echo "GEN_END1 $N rc=$? $(date -Is)"
