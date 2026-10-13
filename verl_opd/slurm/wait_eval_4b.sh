#!/bin/bash
until grep -q "END 4b(resume)" /localscratch/sya225/logs/resume_4b_wrapper.log 2>/dev/null; do sleep 120; done
sleep 90
bash /home/sya225/opd/scripts/post_run.sh opd-1.7b-from4b /localscratch/sya225/checkpoints/opd_code_17b_from4b 0 8030
