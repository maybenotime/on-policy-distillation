#!/bin/bash
# Scan running opd-base jobs for zombie signature, scancel + resubmit.
# Zombie = NCCL "Connection closed by remote peer" in rank-0 log + last 'loss': line
#          older than 10 min (i.e. no training progress).
#
# State file ~/.zombie_resubmit_state.json maps job_id → {args} for replay.
# Max retries per train: 5.

set -u
cd /project/aip-angelx/sya225/on-policy-distillation

NOW=$(date +%s)
STALE_SEC=600  # 10 min without new step → stale

declare -A SEEN

for JOB in $(squeue -u sya225 -h -o '%i %j' | awk '$2=="opd-base" || $2=="opd-base-q3" {print $1}'); do
    F=$(ls logs/*-${JOB}-0.out 2>/dev/null | head -1)
    [ -z "$F" ] && continue
    # NCCL zombie signature
    if grep -q "NCCL WARN socketProgress: Connection closed" "$F" 2>/dev/null; then
        LAST_LOSS_TS=$(grep -E "'loss':" "$F" 2>/dev/null | tail -1 | grep -oE '[0-9]+m [0-9]+s' | head -1)
        # If no 'loss' line at all, def zombie
        if [ -z "$LAST_LOSS_TS" ]; then
            # Get elapsed time of job
            ELAPSED=$(squeue -h -j "$JOB" -o '%M' 2>/dev/null | head -1)
            # Convert HH:MM:SS → seconds
            E_SEC=$(echo "$ELAPSED" | awk -F: '{ if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1 }')
            if [ -n "$E_SEC" ] && [ "$E_SEC" -gt "$STALE_SEC" ]; then
                echo "[zombie] JOB=$JOB elapsed=${ELAPSED} NCCL fail, no step ever"
                # Extract train config from main log
                MAIN_LOG=$(ls logs/*-${JOB}.out 2>/dev/null | head -1)
                STU=$(grep -oE 'student=[^ ]+' "$MAIN_LOG" 2>/dev/null | head -1 | cut -d= -f2)
                TCH=$(grep -oE 'teacher=[^ ]+' "$MAIN_LOG" 2>/dev/null | head -1 | cut -d= -f2)
                DATA=$(grep -oE 'data=[0-9]+' "$MAIN_LOG" 2>/dev/null | head -1 | cut -d= -f2)
                # Output dir gives us OUTPUT_TAG_SUFFIX (after epochs1_)
                OUT_DIR=$(grep -oE 'opd_baseline_(qwen3_)?[^ /]+_teacher_[^ /]+_epochs[0-9]+_[^/ ]+' "$MAIN_LOG" 2>/dev/null | head -1)
                TAG=$(echo "$OUT_DIR" | sed -E 's/.*_epochs[0-9]+_//')
                # OPD_DATASET infer from tag: q2bothcorr7k → q2both_correct_7k
                case "$TAG" in
                    q2bothcorr7k) DS_PATH="../data/prompts_math_swift_q2both_correct_7k.jsonl#${DATA}" ;;
                    q2bothwrong7k) DS_PATH="../data/prompts_math_swift_q2both_wrong_7k.jsonl#${DATA}" ;;
                    t14c3bc2784) DS_PATH="../data/prompts_math_swift_t14corr_3bcorr_2784.jsonl#${DATA}" ;;
                    t14c3bw2784) DS_PATH="../data/prompts_math_swift_t14corr_3bwrong_2784.jsonl#${DATA}" ;;
                    all3w2784) DS_PATH="../data/prompts_math_swift_all3wrong_2784.jsonl#${DATA}" ;;
                    *)
                        echo "[zombie] unknown tag $TAG for JOB=$JOB; skip"
                        continue
                        ;;
                esac
                if [ -z "$STU" ] || [ -z "$TCH" ]; then
                    echo "[zombie] JOB=$JOB couldn't parse student/teacher; skip"
                    continue
                fi
                # Cancel + resubmit
                scancel "$JOB"
                NEW_JOB=$(OPD_DATASET="$DS_PATH" OUTPUT_TAG_SUFFIX="$TAG" \
                    sbatch --parsable slurm_jobs/submit_opd_baseline_vulcan.sh \
                    "$STU" "$TCH" "$DATA" 1)
                echo "[zombie] cancelled $JOB → resubmitted $NEW_JOB (student=$STU teacher=$TCH tag=$TAG)"
            fi
        fi
    fi
done

echo "[zombie-check] done at $(date)"
