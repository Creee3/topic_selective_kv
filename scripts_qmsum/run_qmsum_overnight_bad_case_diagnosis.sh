#!/bin/bash
# ============================================================================
# Diagnose the saved overnight QMSum mainline result without rerunning the model.
#
# Default target:
#   docs=0:30, max_queries=5, overnight_dyn_grounded_96
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_overnight_bad_case_diagnosis.sh
# ============================================================================

set -u

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [ "$SCRIPT_DIR" = "$SCRIPT_PATH" ]; then
    SCRIPT_DIR="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
PYTHON=${PYTHON:-python}

NUM_NODES=${NUM_NODES:-4}
START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-30}
MAX_QUERIES=${MAX_QUERIES:-5}
TARGET_NAME=${TARGET_NAME:-dyn_grounded_96}
TARGET_TAG=${TARGET_TAG:-overnight_dyn_grounded_96_s4_r0.65_m64_k12_batch64_q${MAX_QUERIES}}
ANSWER_MAX_NEW_TOKENS=${ANSWER_MAX_NEW_TOKENS:-96}
TOP_N=${TOP_N:-40}

LOW_RECALL_THRESHOLD=${LOW_RECALL_THRESHOLD:-0.10}
LARGE_GAP_THRESHOLD=${LARGE_GAP_THRESHOLD:-0.10}
LOW_ORACLE_THRESHOLD=${LOW_ORACLE_THRESHOLD:-0.12}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

OUT_DIR=${OUT_DIR:-logs/qmsum_overnight_bad_case_diagnosis_${START_DOC}_${END_DOC}_${TARGET_NAME}}

TARGET_TSV="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${TARGET_TAG}.tsv"
TARGET_LOG="outputs/qmsum_answer_log_N${NUM_NODES}_${START_DOC}_${END_DOC}_${TARGET_TAG}.jsonl"

echo "============================================================"
echo " QMSum Overnight Bad Case Diagnosis"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries=$MAX_QUERIES"
echo " target_name=$TARGET_NAME"
echo " target_tag=$TARGET_TAG"
echo " target_tsv=$TARGET_TSV"
echo " target_log=$TARGET_LOG"
echo " out_dir=$OUT_DIR"
echo "============================================================"

missing=0
for path in "$TARGET_TSV" "$TARGET_LOG"; do
    if [ ! -f "$path" ]; then
        echo "MISSING required file: $path"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo ""
    echo "Please sync the required overnight output files first, then rerun this script."
    exit 1
fi

ANALYZER="$SCRIPT_DIR/analyze_qmsum_bad_cases.py"
if [ ! -f "$ANALYZER" ]; then
    echo "MISSING analyzer script: $ANALYZER"
    echo "Please sync scripts_qmsum/analyze_qmsum_bad_cases.py to the cloud."
    exit 1
fi

"$PYTHON" "$ANALYZER" \
    --target-tsv "$TARGET_TSV" \
    --target-answer-log "$TARGET_LOG" \
    --target-name "$TARGET_NAME" \
    --out-dir "$OUT_DIR" \
    --top-n "$TOP_N" \
    --low-recall-threshold "$LOW_RECALL_THRESHOLD" \
    --large-gap-threshold "$LARGE_GAP_THRESHOLD" \
    --low-oracle-threshold "$LOW_ORACLE_THRESHOLD" \
    --max-model-len "$MAX_MODEL_LEN" \
    --answer-max-new-tokens "$ANSWER_MAX_NEW_TOKENS"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo ""
    echo "Diagnosis failed; no output summary should be trusted."
    exit "$STATUS"
fi

echo ""
echo "Outputs:"
echo "  $OUT_DIR/summary.tsv"
echo "  $OUT_DIR/issue_summary.tsv"
echo "  $OUT_DIR/target_diagnostic_cases.tsv"
echo "  $OUT_DIR/zero_recall_cases.tsv"
echo "  $OUT_DIR/prompt_leak_cases.tsv"
echo "  $OUT_DIR/large_oracle_gap_cases.tsv"
echo "  $OUT_DIR/bad_case_report.md"
