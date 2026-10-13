#!/bin/bash
# full DeepMath labelling: one model, one shard, one GPU.  bash deepmath_full.sh <NAME> <MODEL> <SHARD> <GPU> <FAMILY>
source /home/sya225/opd/env.sh none; PY=$CODE_EVAL_ENV/bin/python; S=$OPD_HOME/scripts; export PATH="$CODE_EVAL_ENV/bin:$PATH" TOKENIZERS_PARALLELISM=false
D=/home/sya225/opd/verl_data/deepmath/sample_tails.parquet; O=$SCRATCH/deepmath; mkdir -p $O
N=$1; M=$2; SH=$3; G=$4; FAM=${5:-ds}
echo "START $N shard $SH gpu $G family $FAM $(date -Is)"
CUDA_VISIBLE_DEVICES=$G $PY $S/deepmath_gen.py --model $M --data $D --shard $SH --nshards ${NSHARDS:-2} --chunk 1500 --max_tokens 16384 --family $FAM \
    --done_from "$O/gen_${N}_shard*.jsonl" --stop_file /home/sya225/opd/verl_data/deepmath/STOP_$N --out $O/gen_${N}_shard$SH.jsonl > $SCRATCH/logs/deepmath_gen_${N}_$SH.log 2>&1
echo "GEN_END $N shard $SH rc=$? $(date -Is)"
