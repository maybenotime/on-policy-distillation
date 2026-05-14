#!/bin/bash -l
# Chain post-train eval jobs onto the 7 in-flight prefix-curriculum / baseline runs.
# Each eval job runs ONLY after its training job exits successfully
# (--dependency=afterok). Single L40S, pinned eval on all ckpts + benchmark on
# the final checkpoint. See submit_post_train_eval_vulcan.sh.
#
# Run this ONCE on the Vulcan login node:
#   cd /project/aip-angelx/sya225/on-policy-distillation
#   bash slurm_jobs/submit_all_post_train_evals.sh
#
# To re-submit one eval job manually, copy the matching sbatch line below.

set -e

cd "$(dirname "$0")/.."
REPO_DIR=$(pwd)
mkdir -p logs

CKPT_BASE=/scratch/sya225/checkpoints

# (TRAIN_JOBID, RUN_DIR, TEACHER, IS_QWEN3)
JOBS=(
  "4958851|${CKPT_BASE}/tp_bpod_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-14B-Instruct_K5_delta0.2_q00.8_filter1_data8910|Qwen/Qwen2.5-14B-Instruct|0"
  "4958852|${CKPT_BASE}/sp_fpod_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-14B-Instruct_K5_delta0.2_p00.2_data8910|Qwen/Qwen2.5-14B-Instruct|0"
  "4958853|${CKPT_BASE}/opd_baseline_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-14B-Instruct_epochs1_data8910|Qwen/Qwen2.5-14B-Instruct|0"
  "4958854|${CKPT_BASE}/opd_baseline_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-14B-Instruct_epochs5_data8910|Qwen/Qwen2.5-14B-Instruct|0"
  "4958855|${CKPT_BASE}/sp_fpod_qwen3_Qwen3-1.7B_teacher_Qwen3-14B_K5_delta0.2_p00.2_data9000|Qwen/Qwen3-14B|1"
  "4958856|${CKPT_BASE}/opd_baseline_qwen3_Qwen3-1.7B_teacher_Qwen3-14B_epochs1_data9000|Qwen/Qwen3-14B|1"
  "4958857|${CKPT_BASE}/opd_baseline_qwen3_Qwen3-1.7B_teacher_Qwen3-14B_epochs5_data9000|Qwen/Qwen3-14B|1"
)

OUT_MAP="${REPO_DIR}/logs/post_eval_jobmap.txt"
: > "$OUT_MAP"

echo "Submitting 7 post-train eval jobs with --dependency=afterok:<train_jobid>"
echo "Job map will be written to $OUT_MAP"
echo ""

for entry in "${JOBS[@]}"; do
    IFS='|' read -r TRAIN_JOBID RUN_DIR TEACHER IS_Q3 <<< "$entry"
    TAG=$(basename "$RUN_DIR")

    OUT=$(sbatch \
        --dependency=afterok:${TRAIN_JOBID} \
        --kill-on-invalid-dep=yes \
        --job-name="eval-${TRAIN_JOBID}" \
        --export=ALL,RUN_DIR="$RUN_DIR",TEACHER="$TEACHER",IS_QWEN3="$IS_Q3" \
        slurm_jobs/submit_post_train_eval_vulcan.sh)

    EVAL_JOBID=$(echo "$OUT" | awk '{print $NF}')
    echo "  train=${TRAIN_JOBID}  eval=${EVAL_JOBID}  qwen3=${IS_Q3}  ${TAG}"
    echo "${TRAIN_JOBID} ${EVAL_JOBID} ${IS_Q3} ${RUN_DIR}" >> "$OUT_MAP"
done

echo ""
echo "Done. squeue should now show 7 jobs in PD with Reason=Dependency."
echo "Verify with: squeue -u \$USER -o '%i %T %R %j'"
