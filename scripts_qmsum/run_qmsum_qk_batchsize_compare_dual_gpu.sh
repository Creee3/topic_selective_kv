#!/bin/bash
# ============================================================================
# Compare two exact-QK batch sizes on two GPUs in parallel.
#
# Goal:
#   test whether batched exact-QK scoring reduces online routing cost without
#   hurting answer quality on the current frozen QMSum mainline.
#
# Default setup:
#   GPU 3 -> QK_SCORE_BATCH_SIZE=8
#   GPU 4 -> QK_SCORE_BATCH_SIZE=32
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_qk_batchsize_compare_dual_gpu.sh
#
# Optional overrides:
#   GPU_A=3 GPU_B=4 BATCH_A=8 BATCH_B=32 \
#   START_DOC=5 END_DOC=10 MAX_QUERIES=5 \
#   bash scripts_qmsum/run_qmsum_qk_batchsize_compare_dual_gpu.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_A=${GPU_A:-3}
GPU_B=${GPU_B:-4}
BATCH_A=${BATCH_A:-8}
BATCH_B=${BATCH_B:-32}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-10}
MAX_QUERIES=${MAX_QUERIES:-5}
NUM_NODES=${NUM_NODES:-4}
ROUTE_TOP_K=${ROUTE_TOP_K:-12}
DYNAMIC_SUMMARY_TOP_K=${DYNAMIC_SUMMARY_TOP_K:-16}
DYNAMIC_DETAIL_TOP_K=${DYNAMIC_DETAIL_TOP_K:-12}
DYNAMIC_BALANCED_TOP_K=${DYNAMIC_BALANCED_TOP_K:-12}
ROUTE_CANDIDATE_PREFILTER=${ROUTE_CANDIDATE_PREFILTER:-lexical}
ROUTE_CANDIDATE_PREFILTER_FACTOR=${ROUTE_CANDIDATE_PREFILTER_FACTOR:-6}
ROUTE_CANDIDATE_PREFILTER_MIN_KEEP=${ROUTE_CANDIDATE_PREFILTER_MIN_KEEP:-48}
ROUTE_CANDIDATE_PREFILTER_MAX_KEEP=${ROUTE_CANDIDATE_PREFILTER_MAX_KEEP:-128}
ANSWER_EVIDENCE_ORDER=${ANSWER_EVIDENCE_ORDER:-qk_then_time}
SELECTED_ANSWER_CONTEXT_MODE=${SELECTED_ANSWER_CONTEXT_MODE:-turns}
ANSWER_PROMPT_STYLE=${ANSWER_PROMPT_STYLE:-strict}
ANSWER_EVIDENCE_MAX_ENTRIES=${ANSWER_EVIDENCE_MAX_ENTRIES:-80}
ANSWER_EVIDENCE_MAX_CHARS=${ANSWER_EVIDENCE_MAX_CHARS:-600}
FETCH_BANDWIDTH_GBPS=${FETCH_BANDWIDTH_GBPS:-25.0}
PER_NODE_RTT_MS=${PER_NODE_RTT_MS:-1.0}
PER_SEGMENT_OVERHEAD_MS=${PER_SEGMENT_OVERHEAD_MS:-0.15}
DECODE_STARTUP_MS=${DECODE_STARTUP_MS:-15.0}
LOG_ROOT=${LOG_ROOT:-logs/qmsum_qk_batchsize_compare_dual_gpu}
LOG_DIR_A="$LOG_ROOT/batch_${BATCH_A}"
LOG_DIR_B="$LOG_ROOT/batch_${BATCH_B}"
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

echo "============================================================"
echo " QMSum QK-BatchSize Compare Dual GPU"
echo " GPU_A=$GPU_A batch=$BATCH_A"
echo " GPU_B=$GPU_B batch=$BATCH_B"
echo " docs=$START_DOC:$END_DOC"
echo " MAX_QUERIES=$MAX_QUERIES"
echo " ROUTE_TOP_K=$ROUTE_TOP_K"
echo " DYNAMIC_DETAIL_TOP_K=$DYNAMIC_DETAIL_TOP_K"
echo " PREFILTER=${ROUTE_CANDIDATE_PREFILTER} factor=$ROUTE_CANDIDATE_PREFILTER_FACTOR min=$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP max=$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP"
echo "============================================================"

GPU_ID="$GPU_A" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
NUM_NODES="$NUM_NODES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
DYNAMIC_ROUTE_BUDGET=1 \
DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
ROUTE_CANDIDATE_PREFILTER_FACTOR="$ROUTE_CANDIDATE_PREFILTER_FACTOR" \
ROUTE_CANDIDATE_PREFILTER_MIN_KEEP="$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP" \
ROUTE_CANDIDATE_PREFILTER_MAX_KEEP="$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP" \
QK_SCORE_BATCH_SIZE="$BATCH_A" \
ANSWER_EVIDENCE_ORDER="$ANSWER_EVIDENCE_ORDER" \
SELECTED_ANSWER_CONTEXT_MODE="$SELECTED_ANSWER_CONTEXT_MODE" \
ANSWER_PROMPT_STYLE="$ANSWER_PROMPT_STYLE" \
ANSWER_EVIDENCE_MAX_ENTRIES="$ANSWER_EVIDENCE_MAX_ENTRIES" \
ANSWER_EVIDENCE_MAX_CHARS="$ANSWER_EVIDENCE_MAX_CHARS" \
FETCH_BANDWIDTH_GBPS="$FETCH_BANDWIDTH_GBPS" \
PER_NODE_RTT_MS="$PER_NODE_RTT_MS" \
PER_SEGMENT_OVERHEAD_MS="$PER_SEGMENT_OVERHEAD_MS" \
DECODE_STARTUP_MS="$DECODE_STARTUP_MS" \
CASE_SUMMARY_TAG="qkbatch_${BATCH_A}_detail${DYNAMIC_DETAIL_TOP_K}_q5" \
LOG_DIR="$LOG_DIR_A" \
bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh" &
PID_A=$!

GPU_ID="$GPU_B" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
NUM_NODES="$NUM_NODES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
DYNAMIC_ROUTE_BUDGET=1 \
DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
ROUTE_CANDIDATE_PREFILTER_FACTOR="$ROUTE_CANDIDATE_PREFILTER_FACTOR" \
ROUTE_CANDIDATE_PREFILTER_MIN_KEEP="$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP" \
ROUTE_CANDIDATE_PREFILTER_MAX_KEEP="$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP" \
QK_SCORE_BATCH_SIZE="$BATCH_B" \
ANSWER_EVIDENCE_ORDER="$ANSWER_EVIDENCE_ORDER" \
SELECTED_ANSWER_CONTEXT_MODE="$SELECTED_ANSWER_CONTEXT_MODE" \
ANSWER_PROMPT_STYLE="$ANSWER_PROMPT_STYLE" \
ANSWER_EVIDENCE_MAX_ENTRIES="$ANSWER_EVIDENCE_MAX_ENTRIES" \
ANSWER_EVIDENCE_MAX_CHARS="$ANSWER_EVIDENCE_MAX_CHARS" \
FETCH_BANDWIDTH_GBPS="$FETCH_BANDWIDTH_GBPS" \
PER_NODE_RTT_MS="$PER_NODE_RTT_MS" \
PER_SEGMENT_OVERHEAD_MS="$PER_SEGMENT_OVERHEAD_MS" \
DECODE_STARTUP_MS="$DECODE_STARTUP_MS" \
CASE_SUMMARY_TAG="qkbatch_${BATCH_B}_detail${DYNAMIC_DETAIL_TOP_K}_q5" \
LOG_DIR="$LOG_DIR_B" \
bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh" &
PID_B=$!

wait $PID_A
STATUS_A=$?

wait $PID_B
STATUS_B=$?

cat > "$SUMMARY_TSV" <<EOF
case	status	qk_batch_size	avg_full_f1	avg_sel_f1	avg_delta	detail_full_f1	detail_sel_f1	detail_delta	avg_candidates_after_prefilter	avg_qk_scoring_ms	avg_routing_ms	avg_selected_ttft_ms	avg_selected_fetch_ms	ctx_token_saving_pct
EOF

append_summary_from_tsv() {
    local case_name="$1"
    local status="$2"
    local batch_size="$3"
    local case_tsv="$4"

    if [ "$status" -ne 0 ] || [ ! -f "$case_tsv" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "FAILED" "$batch_size" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" \
            >> "$SUMMARY_TSV"
        return
    fi

    CASE_NAME="$case_name" BATCH_SIZE="$batch_size" CASE_TSV="$case_tsv" python - <<'PY' >> "$SUMMARY_TSV"
import csv
import os

case_name = os.environ["CASE_NAME"]
batch_size = os.environ["BATCH_SIZE"]
case_tsv = os.environ["CASE_TSV"]

with open(case_tsv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

if not rows:
    print("\t".join([case_name, "EMPTY_TSV", batch_size] + ["FAILED"] * 12))
    raise SystemExit

detail = [r for r in rows if r.get("query_budget_type") == "detail"]

def avg(subset, field):
    return sum(float(r[field]) for r in subset) / len(subset) if subset else 0.0

values = [
    case_name,
    "OK",
    batch_size,
    f"{avg(rows, 'full_answer_f1'):.4f}",
    f"{avg(rows, 'selected_answer_f1'):.4f}",
    f"{avg(rows, 'answer_f1_delta'):.4f}",
    f"{avg(detail, 'full_answer_f1'):.4f}",
    f"{avg(detail, 'selected_answer_f1'):.4f}",
    f"{avg(detail, 'answer_f1_delta'):.4f}",
    f"{avg(rows, 'num_candidates_after_prefilter'):.1f}",
    f"{avg(rows, 'qk_scoring_ms'):.2f}",
    f"{avg(rows, 'routing_overhead_ms'):.2f}",
    f"{avg(rows, 'selected_ttft_ms'):.2f}",
    f"{avg(rows, 'selected_fetch_latency_ms'):.2f}",
    f"{avg(rows, 'ctx_token_saving_pct'):.1f}",
]
print("\t".join(values))
PY
}

TSV_A="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_qkbatch_${BATCH_A}_detail${DYNAMIC_DETAIL_TOP_K}_q5.tsv"
TSV_B="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_qkbatch_${BATCH_B}_detail${DYNAMIC_DETAIL_TOP_K}_q5.tsv"

append_summary_from_tsv "batch_${BATCH_A}" "$STATUS_A" "$BATCH_A" "$TSV_A"
append_summary_from_tsv "batch_${BATCH_B}" "$STATUS_B" "$BATCH_B" "$TSV_B"

{
    echo "============================================================"
    echo " QMSum QK-BatchSize Compare Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Dual GPU compare finished"
echo " GPU_A status=$STATUS_A"
echo " GPU_B status=$STATUS_B"
echo " Logs:"
echo "   $LOG_DIR_A"
echo "   $LOG_DIR_B"
echo " Summary: $SUMMARY_TXT"
echo "============================================================"

if [ "$STATUS_A" -ne 0 ] || [ "$STATUS_B" -ne 0 ]; then
    exit 1
fi
