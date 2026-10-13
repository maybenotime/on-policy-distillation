#!/bin/bash
# job 264964 slot A (GPUs 0-2): math-data OPD, 16k, training only
source /home/USER/opd/scripts/math_common.sh
for t in K4Bmath1200 K4Bcode1200 K4Bcode300; do train $t $(teacher_of $t) 0,1,2; done
echo "CHAIN_MATH_A_DONE $(date -Is)"
