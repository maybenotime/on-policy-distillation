#!/bin/bash
V=/home/USER/opd/envs/verl_opd
P=$("$V/bin/python" -c "import verl,pathlib;print(pathlib.Path(verl.__file__).parent)")
echo "########## trainer/distillation/ ##########"; ls "$P/trainer/distillation/"
echo; echo "########## distillation.yaml ##########"; cat "$P/trainer/config/distillation/distillation.yaml"
echo; echo "########## losses.py (前60行) ##########"; head -60 "$P/trainer/distillation/losses.py"
echo; echo "########## 入口 main ##########"; ls "$P/trainer/"*.py; grep -rl "distillation" "$P/trainer/"*.py 2>/dev/null
