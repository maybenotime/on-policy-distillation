#!/bin/bash
V=/home/USER/opd/envs/verl_opd
P=$("$V/bin/python" -c "import verl,pathlib;print(pathlib.Path(verl.__file__).parent)")
echo "########## distillation.yaml 前50行 ##########"
head -50 "$P/trainer/config/distillation/distillation.yaml"
echo; echo "########## 注册的 loss 名字 ##########"
grep -nE "register_distillation_loss|names\s*=|@register" "$P/trainer/distillation/losses.py" | head -30
