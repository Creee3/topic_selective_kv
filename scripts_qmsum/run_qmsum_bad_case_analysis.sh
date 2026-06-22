#!/bin/bash
# ============================================================================
# Analyze saved QMSum case summaries / answer logs without rerunning the model.
#
# Default target:
#   docs=5:60 baseline vs coarse gate s4_r0.65_m64
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_bad_case_analysis.sh
#
# Optional overrides:
#   START_DOC=5 END_DOC=60 MAX_QUERIES=5 \
#   GATE_TAG=coarsegate_lexical_s4_r0.75_m64_batch64_detail12_q5 \
#   bash scripts_qmsum/run_qmsum_bad_case_analysis.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_NODES=${NUM_NODES:-4}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-60}
MAX_QUERIES=${MAX_QUERIES:-5}
QK_SCORE_BATCH_SIZE=${QK_SCORE_BATCH_SIZE:-64}
DYNAMIC_DETAIL_TOP_K=${DYNAMIC_DETAIL_TOP_K:-12}
TOP_N=${TOP_N:-30}

BASE_TAG=${BASE_TAG:-coarsegate_base_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}}
GATE_TAG=${GATE_TAG:-coarsegate_lexical_s4_r0.65_m64_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}}
OUT_DIR=${OUT_DIR:-logs/qmsum_bad_case_analysis_${START_DOC}_${END_DOC}_${GATE_TAG}}

BASE_PREFIX="outputs/qmsum"
BASE_TSV="${BASE_PREFIX}_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${BASE_TAG}.tsv"
GATE_TSV="${BASE_PREFIX}_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${GATE_TAG}.tsv"
BASE_LOG="${BASE_PREFIX}_answer_log_N${NUM_NODES}_${START_DOC}_${END_DOC}_${BASE_TAG}.jsonl"
GATE_LOG="${BASE_PREFIX}_answer_log_N${NUM_NODES}_${START_DOC}_${END_DOC}_${GATE_TAG}.jsonl"

echo "============================================================"
echo " QMSum Bad Case Analysis"
echo " docs=$START_DOC:$END_DOC"
echo " baseline tag=$BASE_TAG"
echo " gate tag=$GATE_TAG"
echo " out_dir=$OUT_DIR"
echo "============================================================"

missing=0
for path in "$BASE_TSV" "$GATE_TSV"; do
    if [ ! -f "$path" ]; then
        echo "MISSING required file: $path"
        missing=1
    fi
done
for path in "$BASE_LOG" "$GATE_LOG"; do
    if [ ! -f "$path" ]; then
        echo "WARNING optional answer log missing: $path"
    fi
done
if [ "$missing" -ne 0 ]; then
    echo ""
    echo "Please sync the required TSV files first, then rerun this script."
    exit 1
fi

ANALYZER="$SCRIPT_DIR/analyze_qmsum_bad_cases.py"
if [ ! -f "$ANALYZER" ]; then
    echo "MISSING analyzer script: $ANALYZER"
    echo "Please sync scripts_qmsum/analyze_qmsum_bad_cases.py to the cloud."
    exit 1
fi

python "$SCRIPT_DIR/analyze_qmsum_bad_cases.py" \
    --baseline-tsv "$BASE_TSV" \
    --gate-tsv "$GATE_TSV" \
    --baseline-answer-log "$BASE_LOG" \
    --gate-answer-log "$GATE_LOG" \
    --out-dir "$OUT_DIR" \
    --top-n "$TOP_N"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo ""
    echo "Bad case analysis failed; no output summary should be trusted."
    exit "$STATUS"
fi

echo ""
echo "Outputs:"
echo "  $OUT_DIR/summary.tsv"
echo "  $OUT_DIR/worst_selected_vs_full.tsv"
echo "  $OUT_DIR/worst_selected_vs_oracle.tsv"
echo "  $OUT_DIR/gate_regressions_vs_baseline.tsv"
echo "  $OUT_DIR/bad_case_report.md"
