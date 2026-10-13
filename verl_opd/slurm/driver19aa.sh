#!/bin/bash
# Untrained Qwen3-1.7B-Base on all seven comps (8k, MAX_MODEL_LEN 32768, same as its OPD checkpoints), split over 3 GPUs.
# The main tag Q3-1.7B-Base-8k already has AIME24 I/II done from 9-11; MathArena resumes it.
H=/home/sya225/logs; T=/home/sya225/slurm/teacher_lane2.sh; M=Qwen/Qwen3-1.7B-Base
echo "DRIVER19AA_START $(date -Is)"
COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025" bash $T 0 8340 Q3-1.7B-Base-8k:$M:8192:32768  > $H/aa19_0.out 2>&1 &
COMPS="custom/beyondaime"                                                  bash $T 1 8341 Q3-1.7B-Base-8kb:$M:8192:32768 > $H/aa19_1.out 2>&1 &
COMPS="hmmt/hmmt_feb_2026 custom/amc"                                      bash $T 2 8342 Q3-1.7B-Base-8kc:$M:8192:32768 > $H/aa19_2.out 2>&1 &
wait
echo "DRIVER19AA_DONE $(date -Is)"
