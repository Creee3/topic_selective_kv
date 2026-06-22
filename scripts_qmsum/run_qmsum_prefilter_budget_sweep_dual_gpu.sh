#!/bin/bash
# ============================================================================
# Run the candidate-prefilter budget sweep on two GPUs in parallel.
#
# Default split:
#   GPU_A -> smaller / more aggressive prefilter pools
#   GPU_B -> larger / safer prefilter pools including the current baseline
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_prefilter_budget_sweep_dual_gpu.sh
#
# Optional overrides:
#   GPU_A=3 GPU_B=4 START_DOC=5 END_DOC=10 MAX_QUERIES=5 \
#   CASES_A="4:24:64 4:24:96 4:48:96" \
#   CASES_B="6:24:96 6:48:96 6:48:128" \
#   bash scripts_qmsum/run_qmsum_prefilter_budget_sweep_dual_gpu.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_A=${GPU_A:-3}
GPU_B=${GPU_B:-4}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-10}
MAX_QUERIES=${MAX_QUERIES:-5}
NUM_NODES=${NUM_NODES:-4}
ROUTE_TOP_K=${ROUTE_TOP_K:-12}
DYNAMIC_SUMMARY_TOP_K=${DYNAMIC_SUMMARY_TOP_K:-16}
DYNAMIC_DETAIL_TOP_K=${DYNAMIC_DETAIL_TOP_K:-12}
DYNAMIC_BALANCED_TOP_K=${DYNAMIC_BALANCED_TOP_K:-12}
ROUTE_CANDIDATE_PREFILTER=${ROUTE_CANDIDATE_PREFILTER:-lexical}
ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=${ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO:-0.0}
CASES_A=${CASES_A:-"4:24:64 4:24:96 4:48:96"}
CASES_B=${CASES_B:-"6:24:96 6:48:96 6:48:128"}
FETCH_BANDWIDTH_GBPS=${FETCH_BANDWIDTH_GBPS:-25.0}
PER_NODE_RTT_MS=${PER_NODE_RTT_MS:-1.0}
PER_SEGMENT_OVERHEAD_MS=${PER_SEGMENT_OVERHEAD_MS:-0.15}
DECODE_STARTUP_MS=${DECODE_STARTUP_MS:-15.0}
LOG_ROOT=${LOG_ROOT:-logs/qmsum_prefilter_budget_sweep_dual_gpu}
LOG_ROOT_A="$LOG_ROOT/gpu${GPU_A}"
LOG_ROOT_B="$LOG_ROOT/gpu${GPU_B}"
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

echo "============================================================"
echo " QMSum Prefilter-Budget Sweep Dual GPU"
echo " GPU_A=$GPU_A cases=[$CASES_A]"
echo " GPU_B=$GPU_B cases=[$CASES_B]"
echo " docs=$START_DOC:$END_DOC"
echo " MAX_QUERIES=$MAX_QUERIES"
echo " ROUTE_TOP_K=$ROUTE_TOP_K"
echo " DYNAMIC_DETAIL_TOP_K=$DYNAMIC_DETAIL_TOP_K"
echo " ROUTE_CANDIDATE_PREFILTER=$ROUTE_CANDIDATE_PREFILTER"
echo " ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO"
echo "============================================================"

GPU_ID="$GPU_A" \
NUM_NODES="$NUM_NODES" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO="$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO" \
PREFILTER_CASES="$CASES_A" \
FETCH_BANDWIDTH_GBPS="$FETCH_BANDWIDTH_GBPS" \
PER_NODE_RTT_MS="$PER_NODE_RTT_MS" \
PER_SEGMENT_OVERHEAD_MS="$PER_SEGMENT_OVERHEAD_MS" \
DECODE_STARTUP_MS="$DECODE_STARTUP_MS" \
LOG_ROOT="$LOG_ROOT_A" \
bash "$SCRIPT_DIR/run_qmsum_prefilter_budget_sweep.sh" &
PID_A=$!

GPU_ID="$GPU_B" \
NUM_NODES="$NUM_NODES" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO="$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO" \
PREFILTER_CASES="$CASES_B" \
FETCH_BANDWIDTH_GBPS="$FETCH_BANDWIDTH_GBPS" \
PER_NODE_RTT_MS="$PER_NODE_RTT_MS" \
PER_SEGMENT_OVERHEAD_MS="$PER_SEGMENT_OVERHEAD_MS" \
DECODE_STARTUP_MS="$DECODE_STARTUP_MS" \
LOG_ROOT="$LOG_ROOT_B" \
bash "$SCRIPT_DIR/run_qmsum_prefilter_budget_sweep.sh" &
PID_B=$!

wait $PID_A
STATUS_A=$?

wait $PID_B
STATUS_B=$?

if [ -f "$LOG_ROOT_A/summary.tsv" ]; then
    head -n 1 "$LOG_ROOT_A/summary.tsv" > "$SUMMARY_TSV"
    tail -n +2 "$LOG_ROOT_A/summary.tsv" >> "$SUMMARY_TSV"
else
    echo "case	status	prefilter_mode	min_prune_ratio	factor	min_keep	max_keep	avg_full_f1	avg_sel_f1	avg_delta	detail_full_f1	detail_sel_f1	detail_delta	detail_recall	detail_precision	detail_ctx_save_pct	avg_candidates_before	avg_prefilter_pool	avg_candidates_after	avg_prefilter_ms	avg_qk_ms	avg_routing_ms	avg_selected_ttft_ms	avg_selected_fetch_ms	overall_top1_hit_pct	detail_top1_hit_pct" > "$SUMMARY_TSV"
fi

if [ -f "$LOG_ROOT_B/summary.tsv" ]; then
    tail -n +2 "$LOG_ROOT_B/summary.tsv" >> "$SUMMARY_TSV"
fi

{
    echo "============================================================"
    echo " QMSum Prefilter-Budget Sweep Dual-GPU Summary"
    echo "============================================================"
    if [ -f "$SUMMARY_TSV" ]; then
        if command -v column >/dev/null 2>&1; then
            column -t -s $'\t' "$SUMMARY_TSV"
        else
            cat "$SUMMARY_TSV"
        fi
    else
        echo "summary missing"
    fi
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Dual GPU sweep finished"
echo " GPU_A status=$STATUS_A"
echo " GPU_B status=$STATUS_B"
echo " Logs:"
echo "   $LOG_ROOT_A"
echo "   $LOG_ROOT_B"
echo " Combined summary: $SUMMARY_TXT"
echo "============================================================"

if [ "$STATUS_A" -ne 0 ] || [ "$STATUS_B" -ne 0 ]; then
    exit 1
fi
