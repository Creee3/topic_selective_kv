#!/bin/bash
# ============================================================================
# Compare old always-run lexical prefilter vs adaptive low-prune skip.
#
# Default split:
#   GPU_OLD -> min_prune_ratio=0.0, equivalent to the old behavior
#   GPU_NEW -> min_prune_ratio=0.2, current adaptive candidate-skip rule
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_prefilter_minprune_compare_dual_gpu.sh
#
# Optional overrides:
#   START_DOC=5 END_DOC=20 MAX_QUERIES=5 GPU_OLD=0 GPU_NEW=1 CASES="6:48:128" \
#   bash scripts_qmsum/run_qmsum_prefilter_minprune_compare_dual_gpu.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_OLD=${GPU_OLD:-0}
GPU_NEW=${GPU_NEW:-1}

START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-20}
MAX_QUERIES=${MAX_QUERIES:-5}
NUM_NODES=${NUM_NODES:-4}
ROUTE_TOP_K=${ROUTE_TOP_K:-12}
DYNAMIC_SUMMARY_TOP_K=${DYNAMIC_SUMMARY_TOP_K:-16}
DYNAMIC_DETAIL_TOP_K=${DYNAMIC_DETAIL_TOP_K:-12}
DYNAMIC_BALANCED_TOP_K=${DYNAMIC_BALANCED_TOP_K:-12}
ROUTE_CANDIDATE_PREFILTER=${ROUTE_CANDIDATE_PREFILTER:-lexical}
CASES=${CASES:-"6:48:128"}

FETCH_BANDWIDTH_GBPS=${FETCH_BANDWIDTH_GBPS:-25.0}
PER_NODE_RTT_MS=${PER_NODE_RTT_MS:-1.0}
PER_SEGMENT_OVERHEAD_MS=${PER_SEGMENT_OVERHEAD_MS:-0.15}
DECODE_STARTUP_MS=${DECODE_STARTUP_MS:-15.0}

LOG_ROOT=${LOG_ROOT:-logs/qmsum_prefilter_minprune_compare_dual_gpu}
OLD_LOG_ROOT="$LOG_ROOT/minprune0_old"
NEW_LOG_ROOT="$LOG_ROOT/minprune02_adaptive"
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

echo "============================================================"
echo " QMSum Prefilter Min-Prune Compare Dual GPU"
echo " docs=$START_DOC:$END_DOC  MAX_QUERIES=$MAX_QUERIES"
echo " cases=[$CASES]"
echo " old: GPU $GPU_OLD  min_prune_ratio=0.0"
echo " new: GPU $GPU_NEW  min_prune_ratio=0.2"
echo " LOG_ROOT=$LOG_ROOT"
echo "============================================================"

GPU_ID="$GPU_OLD" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
NUM_NODES="$NUM_NODES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=0.0 \
PREFILTER_CASES="$CASES" \
CASE_SUMMARY_TAG_SUFFIX="_minprune0" \
FETCH_BANDWIDTH_GBPS="$FETCH_BANDWIDTH_GBPS" \
PER_NODE_RTT_MS="$PER_NODE_RTT_MS" \
PER_SEGMENT_OVERHEAD_MS="$PER_SEGMENT_OVERHEAD_MS" \
DECODE_STARTUP_MS="$DECODE_STARTUP_MS" \
LOG_ROOT="$OLD_LOG_ROOT" \
bash "$SCRIPT_DIR/run_qmsum_prefilter_budget_sweep.sh" &
PID_OLD=$!

GPU_ID="$GPU_NEW" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
NUM_NODES="$NUM_NODES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=0.2 \
PREFILTER_CASES="$CASES" \
CASE_SUMMARY_TAG_SUFFIX="_minprune02" \
FETCH_BANDWIDTH_GBPS="$FETCH_BANDWIDTH_GBPS" \
PER_NODE_RTT_MS="$PER_NODE_RTT_MS" \
PER_SEGMENT_OVERHEAD_MS="$PER_SEGMENT_OVERHEAD_MS" \
DECODE_STARTUP_MS="$DECODE_STARTUP_MS" \
LOG_ROOT="$NEW_LOG_ROOT" \
bash "$SCRIPT_DIR/run_qmsum_prefilter_budget_sweep.sh" &
PID_NEW=$!

wait $PID_OLD
STATUS_OLD=$?

wait $PID_NEW
STATUS_NEW=$?

if [ -f "$OLD_LOG_ROOT/summary.tsv" ]; then
    head -n 1 "$OLD_LOG_ROOT/summary.tsv" > "$SUMMARY_TSV"
    tail -n +2 "$OLD_LOG_ROOT/summary.tsv" >> "$SUMMARY_TSV"
elif [ -f "$NEW_LOG_ROOT/summary.tsv" ]; then
    head -n 1 "$NEW_LOG_ROOT/summary.tsv" > "$SUMMARY_TSV"
else
    echo "case	status	prefilter_mode	min_prune_ratio	factor	min_keep	max_keep	avg_full_f1	avg_sel_f1	avg_delta	detail_full_f1	detail_sel_f1	detail_delta	detail_recall	detail_precision	detail_ctx_save_pct	avg_candidates_before	avg_prefilter_pool	avg_candidates_after	avg_prefilter_ms	avg_qk_ms	avg_routing_ms	avg_selected_ttft_ms	avg_selected_fetch_ms	overall_top1_hit_pct	detail_top1_hit_pct" > "$SUMMARY_TSV"
fi

if [ -f "$NEW_LOG_ROOT/summary.tsv" ]; then
    tail -n +2 "$NEW_LOG_ROOT/summary.tsv" >> "$SUMMARY_TSV"
fi

{
    echo "============================================================"
    echo " QMSum Prefilter Min-Prune Compare Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
    echo ""
    echo "Old logs: $OLD_LOG_ROOT"
    echo "New logs: $NEW_LOG_ROOT"
    echo "Combined summary: $SUMMARY_TSV"
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Compare finished"
echo " old status=$STATUS_OLD"
echo " new status=$STATUS_NEW"
echo " summary=$SUMMARY_TXT"
echo "============================================================"

if [ "$STATUS_OLD" -ne 0 ] || [ "$STATUS_NEW" -ne 0 ]; then
    exit 1
fi
