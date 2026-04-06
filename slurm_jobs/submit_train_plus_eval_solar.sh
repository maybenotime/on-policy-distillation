#!/bin/bash

cd /home/sya225/on-policy-distillation/slurm_jobs

JOB_ID=$(sbatch --parsable submit_on_policy_distillation_solar.sh)
echo "Training job: $JOB_ID"

sbatch --dependency=afterany:$JOB_ID --export=ALL,TRAIN_JOB_ID=$JOB_ID eval_solar.sh
echo "Eval job submitted, waiting for training job $JOB_ID"
