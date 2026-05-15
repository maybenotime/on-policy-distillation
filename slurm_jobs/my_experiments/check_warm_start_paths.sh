#!/bin/bash
# check_warm_start_paths.sh
# Verify AT-SFT checkpoint paths used by opsd_warm_start_{0.5b,3b}_fir.sh.
# Run interactively on the cluster before submitting jobs.
#
# Usage:
#   bash check_warm_start_paths.sh

BASE="${BASE:-/home/cta156/scratch/on-policy-distillation}"
AT_SFT_ROOT_3B="${AT_SFT_ROOT:-${BASE}/output}"

PASS=0
FAIL=0

check() {
    local label="$1"
    local path="$2"
    if [ -d "$path" ]; then
        local ckpt
        ckpt=$(find "$path" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
        if [ -n "$ckpt" ]; then
            printf "  ✓  %-55s → %s\n" "$label" "${ckpt##*/}"
            PASS=$((PASS + 1))
        else
            printf "  ✗  %-55s dir exists but NO checkpoint-* found\n" "$label"
            FAIL=$((FAIL + 1))
        fi
    else
        printf "  ✗  %-55s NOT FOUND: %s\n" "$label" "$path"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo "0.5B AT-SFT checkpoints"
echo "  base: \$BASE/output/at_sft_Qwen2.5-0.5B-Instruct_ratio{30,50,70}_data5000"
echo "========================================"
for ratio in 30 50 70; do
    check "0.5B ratio${ratio}" "${BASE}/output/at_sft_Qwen2.5-0.5B-Instruct_ratio${ratio}_data5000"
done

echo ""
echo "========================================"
echo "3B AT-SFT checkpoints"
echo "  base: ${AT_SFT_ROOT_3B}/at_sft_Qwen2.5-3B-Instruct_ratio{10,30,50,70,90}_data5000"
echo "========================================"
for ratio in 10 30 50 70 90; do
    check "3B ratio${ratio}" "${AT_SFT_ROOT_3B}/at_sft_Qwen2.5-3B-Instruct_ratio${ratio}_data5000"
done


echo ""
echo "========================================"
echo "OPSD data file"
echo "========================================"
OPSD_DATA="${BASE}/data/math_all_teachers_correct.jsonl"
if [ -f "$OPSD_DATA" ]; then
    lines=$(wc -l < "$OPSD_DATA")
    printf "  ✓  math_all_teachers_correct.jsonl  (%s lines)\n" "$lines"
    PASS=$((PASS + 1))
else
    printf "  ✗  NOT FOUND: %s\n" "$OPSD_DATA"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "========================================"
printf "Result: %d passed, %d failed\n" "$PASS" "$FAIL"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Fix failed paths before submitting jobs."
    echo "Override path variables if needed:"
    echo "  BASE=/your/path bash check_warm_start_paths.sh"
    echo "  AT_SFT_ROOT=/your/3b/path bash check_warm_start_paths.sh"
    exit 1
fi
