#!/bin/bash
source /home/sya225/opd/env.sh none; P=$CODE_EVAL_ENV/bin/python; D=/localscratch/sya225/fork_topk; F=/home/sya225/opd/fork; S=/home/sya225/opd/scripts/residual_diag.py
TK=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
$P $S --student $D/preds_TJustRL_student.npz --teacher $D/preds_TJustRL_teacher.npz --rollouts $F/roll_ds15b-dapo.jsonl --tok $TK --label "PRE  DS<-JustRL"
$P $S --student $D/preds_post_JustRL_student.npz --teacher $D/preds_post_JustRL_teacher.npz --rollouts $F/roll_ds15b-fromJustRL-post.jsonl --tok $TK --label "POST DS<-JustRL"
$P $S --student $D/preds_TJustRL_student.npz --teacher $D/preds_TSky7B.npz --rollouts $F/roll_ds15b-dapo.jsonl --tok $TK --label "PRE  DS<-Sky7B"
$P $S --student $D/preds_post_Sky7B_student.npz --teacher $D/preds_post_Sky7B_teacher.npz --rollouts $F/roll_ds15b-fromSky7B-post.jsonl --tok $TK --label "POST DS<-Sky7B"
$P $S --student $D/preds_TJustRL_student.npz --teacher $D/preds_TDS7B.npz --rollouts $F/roll_ds15b-dapo.jsonl --tok $TK --label "PRE  DS<-DS7B"
$P $S --student $D/preds_post_DS7B_student.npz --teacher $D/preds_post_DS7B_teacher.npz --rollouts $F/roll_ds15b-fromDS7B-post.jsonl --tok $TK --label "POST DS<-DS7B"
echo "RESID_DONE $(date -Is)"
