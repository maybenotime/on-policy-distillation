#!/bin/bash
# run 3 (Qwen3-4B prefs ⊕ RL-Math shape) step 100 on the seven comps, 8k, split over 3 GPUs.
H=/home/sya225/logs; T=/home/sya225/slurm/teacher_lane2.sh; M=/localscratch/sya225/checkpoints/hf_q3i17_tpA4B_shapeM500_step100
echo "DRIVER19AB_START $(date -Is)"
COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025" bash $T 0 8350 q3TP_A4B_shapeM500-s100-16k:$M:8192  > $H/ab19_0.out 2>&1 &
COMPS="custom/beyondaime"                                                  bash $T 1 8351 q3TP_A4B_shapeM500-s100b-16k:$M:8192 > $H/ab19_1.out 2>&1 &
COMPS="hmmt/hmmt_feb_2026 custom/amc"                                      bash $T 2 8352 q3TP_A4B_shapeM500-s100c-16k:$M:8192 > $H/ab19_2.out 2>&1 &
wait
echo "DRIVER19AB_DONE $(date -Is)"
