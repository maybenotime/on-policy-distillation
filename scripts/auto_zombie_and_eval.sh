#!/bin/bash
# Combined: detect zombies + submit bench eval for newly-completed trains.
# Tracks state via ~/auto_pipeline_state.txt to avoid double-submission.
set -u
cd /project/aip-angelx/sya225/on-policy-distillation
STATE_FILE="${HOME}/auto_pipeline_state.txt"
touch "$STATE_FILE"

# === Part 1: zombie detection (reuse existing logic) ===
for JOB in $(squeue -u sya225 -h -o '%i %j' | awk '$2=="opd-base" || $2=="opd-base-q3" {print $1}'); do
    F=$(ls logs/*-${JOB}-0.out 2>/dev/null | head -1)
    [ -z "$F" ] && continue
    if grep -q "NCCL WARN socketProgress: Connection closed" "$F" 2>/dev/null; then
        LAST_LOSS=$(grep -E "'loss':" "$F" 2>/dev/null | tail -1)
        if [ -z "$LAST_LOSS" ]; then
            ELAPSED=$(squeue -h -j "$JOB" -o '%M' 2>/dev/null | head -1)
            E_SEC=$(echo "$ELAPSED" | awk -F: '{ if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1 }')
            if [ -n "$E_SEC" ] && [ "$E_SEC" -gt 600 ]; then
                echo "[zombie] JOB=$JOB elapsed=$ELAPSED NCCL fail"
                MAIN_LOG=$(ls logs/*-${JOB}.out 2>/dev/null | head -1)
                STU=$(grep -oE 'student=[^ ]+' "$MAIN_LOG" 2>/dev/null | head -1 | cut -d= -f2)
                TCH=$(grep -oE 'teacher=[^ ]+' "$MAIN_LOG" 2>/dev/null | head -1 | cut -d= -f2)
                DATA=$(grep -oE 'data=[0-9]+' "$MAIN_LOG" 2>/dev/null | head -1 | cut -d= -f2)
                OUT_DIR=$(grep -oE 'opd_baseline_(qwen3_)?[^ /]+_teacher_[^ /]+_epochs[0-9]+_[^/ ]+' "$MAIN_LOG" 2>/dev/null | head -1)
                TAG=$(echo "$OUT_DIR" | sed -E 's/.*_epochs[0-9]+_//')
                case "$TAG" in
                    q2bothcorr7k) DS_PATH="../data/prompts_math_swift_q2both_correct_7k.jsonl#${DATA}" ;;
                    q2bothwrong7k) DS_PATH="../data/prompts_math_swift_q2both_wrong_7k.jsonl#${DATA}" ;;
                    t14c3bc2784) DS_PATH="../data/prompts_math_swift_t14corr_3bcorr_2784.jsonl#${DATA}" ;;
                    t14c3bw2784) DS_PATH="../data/prompts_math_swift_t14corr_3bwrong_2784.jsonl#${DATA}" ;;
                    all3w2784) DS_PATH="../data/prompts_math_swift_all3wrong_2784.jsonl#${DATA}" ;;
                    *) echo "[zombie] unknown tag $TAG; skip"; continue ;;
                esac
                if [ -z "$STU" ] || [ -z "$TCH" ]; then continue; fi
                scancel "$JOB"
                NEW=$(OPD_DATASET="$DS_PATH" OUTPUT_TAG_SUFFIX="$TAG" \
                    sbatch --parsable slurm_jobs/submit_opd_baseline_vulcan.sh "$STU" "$TCH" "$DATA" 1)
                echo "[zombie] $JOB → $NEW (stu=$STU tch=$TCH tag=$TAG)"
            fi
        fi
    fi
done

# === Part 2: auto-submit bench eval for completed trains ===
# Scan recent train job IDs; if COMPLETED and not in state file, submit bench.
for JOB in $(sacct -u sya225 -X --noheader -o JobID,State,JobName -S "$(date -d '1 day ago' +%Y-%m-%dT%H:%M:%S)" 2>/dev/null \
              | awk '$2=="COMPLETED" && ($3=="opd-base" || $3=="opd-base-q3") {print $1}'); do
    if grep -q "^EVAL_SUBMITTED:${JOB}:" "$STATE_FILE"; then continue; fi
    # Find the ckpt path from this job's log
    MAIN_LOG=$(ls logs/*-${JOB}.out 2>/dev/null | head -1)
    [ -z "$MAIN_LOG" ] && continue
    CKPT=$(grep -E "Checkpoint saved by trap|Checkpoint saved:" "$MAIN_LOG" 2>/dev/null | tail -1 | grep -oE '/scratch[^ ]+/checkpoint-[0-9]+')
    [ -z "$CKPT" ] && continue
    RUN_DIR=$(dirname "$CKPT")
    # Derive tag for naming the bench eval run
    TAG_HUMAN=$(echo "$RUN_DIR" | grep -oE '/opd_baseline_(qwen3_)?[^/]+' | sed 's|/opd_baseline_||' | sed 's/_epochs1_/+/' | sed 's/_teacher_/__/')
    # Submit bench eval
    NEW_EVAL=$(RUN_DIR="$RUN_DIR" RUN_TAG="$TAG_HUMAN" \
        sbatch --parsable slurm_jobs/submit_benchmark_eval_vulcan.sh)
    echo "[bench-eval] train $JOB COMPLETED → submit eval $NEW_EVAL (run=$TAG_HUMAN)"
    echo "EVAL_SUBMITTED:${JOB}:${NEW_EVAL}:${RUN_DIR}" >> "$STATE_FILE"
done

echo "[pipeline] done at $(date)"
