#!/bin/bash
# Fan the five models across five GPUs, one each; card 5 stays free as headroom.
set -x
W=/home/USER/opd/scripts/eval_one_model.sh
L=/localscratch/USER/logs
run () { bash "$W" "$@" > "$L/evalall_$3.log" 2>&1 & echo "started $3 pid=$!"; }

#     cfg          model                                              tag         port  gpu
run qwen3_8b     Qwen/Qwen3-8B                                        Qwen3-8B    8010  0
run qwen3_4b     Qwen/Qwen3-4B                                        Qwen3-4B    8011  1
run qwen3_1p7b   Qwen/Qwen3-1.7B                                      Qwen3-1.7B  8012  2
run opd_4b       /localscratch/USER/checkpoints/hf_opd30k_step424    opd-4b      8013  3
run opd_1p7b     /localscratch/USER/checkpoints/hf_opd17b_step424    opd-1.7b    8014  4
wait
echo "ALL_MODELS_DONE $(date -Is)"
