#!/bin/bash
EX=/home/sya225/opd/ckpt_export/opd_code_17b_from4b
until [ -f $EX/EXPORT_DONE ]; do sleep 60; done
echo "export seen $(date -Is)"; sleep 30
bash /home/sya225/opd/scripts/post_run.sh opd-1.7b-from4b $EX 1 8030
echo "WAIT_EVAL_4B_19_DONE rc=$?"
