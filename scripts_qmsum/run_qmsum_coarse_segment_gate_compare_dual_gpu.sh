#!/bin/bash
# ============================================================================
# Compare baseline exact Q-K candidate set vs lexical coarse-segment gate.
#
# Default split:
#   GPU_BASE -> baseline, no coarse segment gate
#   GPU_GATE -> route_coarse_segment_gate=lexical
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_coarse_segment_gate_compare_dual_gpu.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_BASE=${GPU_BASE:-0}
GPU_GATE=${GPU_GATE:-1}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-20}
MAX_QUERIES=${MAX_QUERIES:-5}
NUM_NODES=${NUM_NODES:-4}

ROUTE_TOP_K=${ROUTE_TOP_K:-12}
QK_SCORE_BATCH_SIZE=${QK_SCORE_BATCH_SIZE:-64}
CACHE_CANDIDATE_KEYS=${CACHE_CANDIDATE_KEYS:-1}

DYNAMIC_SUMMARY_TOP_K=${DYNAMIC_SUMMARY_TOP_K:-16}
DYNAMIC_DETAIL_TOP_K=${DYNAMIC_DETAIL_TOP_K:-12}
DYNAMIC_BALANCED_TOP_K=${DYNAMIC_BALANCED_TOP_K:-12}

ROUTE_CANDIDATE_PREFILTER=${ROUTE_CANDIDATE_PREFILTER:-lexical}
ROUTE_CANDIDATE_PREFILTER_FACTOR=${ROUTE_CANDIDATE_PREFILTER_FACTOR:-6}
ROUTE_CANDIDATE_PREFILTER_MIN_KEEP=${ROUTE_CANDIDATE_PREFILTER_MIN_KEEP:-48}
ROUTE_CANDIDATE_PREFILTER_MAX_KEEP=${ROUTE_CANDIDATE_PREFILTER_MAX_KEEP:-128}
ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=${ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO:-0.0}

ROUTE_COARSE_SEGMENT_SIZE=${ROUTE_COARSE_SEGMENT_SIZE:-4}
ROUTE_COARSE_SEGMENT_KEEP_RATIO=${ROUTE_COARSE_SEGMENT_KEEP_RATIO:-0.5}
ROUTE_COARSE_SEGMENT_MIN_KEEP=${ROUTE_COARSE_SEGMENT_MIN_KEEP:-48}
ROUTE_COARSE_SEGMENT_MAX_KEEP=${ROUTE_COARSE_SEGMENT_MAX_KEEP:-0}

ANSWER_EVIDENCE_ORDER=${ANSWER_EVIDENCE_ORDER:-qk_then_time}
SELECTED_ANSWER_CONTEXT_MODE=${SELECTED_ANSWER_CONTEXT_MODE:-turns}
ANSWER_PROMPT_STYLE=${ANSWER_PROMPT_STYLE:-strict}
ANSWER_EVIDENCE_MAX_ENTRIES=${ANSWER_EVIDENCE_MAX_ENTRIES:-80}
ANSWER_EVIDENCE_MAX_CHARS=${ANSWER_EVIDENCE_MAX_CHARS:-600}

FETCH_BANDWIDTH_GBPS=${FETCH_BANDWIDTH_GBPS:-25.0}
PER_NODE_RTT_MS=${PER_NODE_RTT_MS:-1.0}
PER_SEGMENT_OVERHEAD_MS=${PER_SEGMENT_OVERHEAD_MS:-0.15}
DECODE_STARTUP_MS=${DECODE_STARTUP_MS:-15.0}
QUERY_TOKENIZER_WARMUP=${QUERY_TOKENIZER_WARMUP:-1}

LOG_ROOT=${LOG_ROOT:-logs/qmsum_coarse_segment_gate_compare_dual_gpu}
BASE_LOG_ROOT="$LOG_ROOT/baseline"
GATE_LOG_ROOT="$LOG_ROOT/coarse_gate"
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

run_case() {
    local gpu_id="$1"
    local gate_mode="$2"
    local tag="$3"
    local log_dir="$4"

    GPU_ID="$gpu_id" \
    START_DOC="$START_DOC" \
    END_DOC="$END_DOC" \
    MAX_QUERIES="$MAX_QUERIES" \
    NUM_NODES="$NUM_NODES" \
    ROUTE_TOP_K="$ROUTE_TOP_K" \
    QK_SCORE_BATCH_SIZE="$QK_SCORE_BATCH_SIZE" \
    CACHE_CANDIDATE_KEYS="$CACHE_CANDIDATE_KEYS" \
    DYNAMIC_ROUTE_BUDGET=1 \
    DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
    DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
    DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
    ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
    ROUTE_CANDIDATE_PREFILTER_FACTOR="$ROUTE_CANDIDATE_PREFILTER_FACTOR" \
    ROUTE_CANDIDATE_PREFILTER_MIN_KEEP="$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP" \
    ROUTE_CANDIDATE_PREFILTER_MAX_KEEP="$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP" \
    ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO="$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO" \
    ROUTE_COARSE_SEGMENT_GATE="$gate_mode" \
    ROUTE_COARSE_SEGMENT_SIZE="$ROUTE_COARSE_SEGMENT_SIZE" \
    ROUTE_COARSE_SEGMENT_KEEP_RATIO="$ROUTE_COARSE_SEGMENT_KEEP_RATIO" \
    ROUTE_COARSE_SEGMENT_MIN_KEEP="$ROUTE_COARSE_SEGMENT_MIN_KEEP" \
    ROUTE_COARSE_SEGMENT_MAX_KEEP="$ROUTE_COARSE_SEGMENT_MAX_KEEP" \
    ANSWER_EVIDENCE_ORDER="$ANSWER_EVIDENCE_ORDER" \
    SELECTED_ANSWER_CONTEXT_MODE="$SELECTED_ANSWER_CONTEXT_MODE" \
    ANSWER_PROMPT_STYLE="$ANSWER_PROMPT_STYLE" \
    ANSWER_EVIDENCE_MAX_ENTRIES="$ANSWER_EVIDENCE_MAX_ENTRIES" \
    ANSWER_EVIDENCE_MAX_CHARS="$ANSWER_EVIDENCE_MAX_CHARS" \
    FETCH_BANDWIDTH_GBPS="$FETCH_BANDWIDTH_GBPS" \
    PER_NODE_RTT_MS="$PER_NODE_RTT_MS" \
    PER_SEGMENT_OVERHEAD_MS="$PER_SEGMENT_OVERHEAD_MS" \
    DECODE_STARTUP_MS="$DECODE_STARTUP_MS" \
    QUERY_TOKENIZER_WARMUP="$QUERY_TOKENIZER_WARMUP" \
    CASE_SUMMARY_TAG="$tag" \
    LOG_DIR="$log_dir" \
    RESUME_IF_LOG_OK=0 \
    bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh" &
}

echo "============================================================"
echo " QMSum Coarse Segment Gate Compare Dual GPU"
echo " docs=$START_DOC:$END_DOC  MAX_QUERIES=$MAX_QUERIES"
echo " baseline GPU=$GPU_BASE"
echo " coarse gate GPU=$GPU_GATE"
echo " gate size=$ROUTE_COARSE_SEGMENT_SIZE keep_ratio=$ROUTE_COARSE_SEGMENT_KEEP_RATIO min_keep=$ROUTE_COARSE_SEGMENT_MIN_KEEP max_keep=$ROUTE_COARSE_SEGMENT_MAX_KEEP"
echo " LOG_ROOT=$LOG_ROOT"
echo "============================================================"

BASE_TAG="coarsegate_base_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}"
GATE_TAG="coarsegate_lexical_s${ROUTE_COARSE_SEGMENT_SIZE}_r${ROUTE_COARSE_SEGMENT_KEEP_RATIO}_m${ROUTE_COARSE_SEGMENT_MIN_KEEP}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}"

run_case "$GPU_BASE" "none" "$BASE_TAG" "$BASE_LOG_ROOT"
PID_BASE=$!

run_case "$GPU_GATE" "lexical" "$GATE_TAG" "$GATE_LOG_ROOT"
PID_GATE=$!

wait $PID_BASE
STATUS_BASE=$?

wait $PID_GATE
STATUS_GATE=$?

BASE_TSV="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${BASE_TAG}.tsv"
GATE_TSV="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${GATE_TAG}.tsv"

cat > "$SUMMARY_TSV" <<EOF
case	status	gate_mode	avg_full_f1	avg_sel_f1	avg_delta	avg_turn_recall	avg_turn_precision	avg_turn_f1	avg_candidates_before	avg_candidates_after	avg_gate_before	avg_gate_after	avg_gate_prune_pct	avg_gate_ms	avg_qk_ms	avg_routing_ms	avg_selected_ttft_ms	ctx_token_saving_pct
EOF

append_summary() {
    local case_name="$1"
    local status="$2"
    local gate_mode="$3"
    local case_tsv="$4"

    if [ "$status" -ne 0 ] || [ ! -f "$case_tsv" ]; then
        echo "$case_name	RUN_FAIL	$gate_mode	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED" >> "$SUMMARY_TSV"
        return
    fi

    CASE_NAME="$case_name" GATE_MODE="$gate_mode" CASE_TSV="$case_tsv" python - <<'PY' >> "$SUMMARY_TSV"
import csv
import os

case_name = os.environ["CASE_NAME"]
gate_mode = os.environ["GATE_MODE"]
case_tsv = os.environ["CASE_TSV"]

with open(case_tsv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

def avg(field):
    vals = []
    for row in rows:
        value = row.get(field, "")
        if value in ["", None]:
            continue
        vals.append(float(value))
    return sum(vals) / len(vals) if vals else 0.0

values = [
    case_name,
    "OK",
    gate_mode,
    f"{avg('full_answer_f1'):.4f}",
    f"{avg('selected_answer_f1'):.4f}",
    f"{avg('answer_f1_delta'):.4f}",
    f"{avg('selected_turn_recall'):.4f}",
    f"{avg('selected_turn_precision'):.4f}",
    f"{avg('selected_turn_f1'):.4f}",
    f"{avg('num_candidates_before_prefilter'):.1f}",
    f"{avg('num_candidates_after_prefilter'):.1f}",
    f"{avg('coarse_segment_gate_before'):.1f}",
    f"{avg('coarse_segment_gate_after'):.1f}",
    f"{100.0 * avg('coarse_segment_gate_prune_ratio'):.1f}",
    f"{avg('coarse_segment_gate_ms'):.2f}",
    f"{avg('qk_scoring_ms'):.2f}",
    f"{avg('routing_overhead_ms'):.2f}",
    f"{avg('selected_ttft_ms'):.2f}",
    f"{avg('ctx_token_saving_pct'):.1f}",
]
print("\t".join(values))
PY
}

append_summary "baseline" "$STATUS_BASE" "none" "$BASE_TSV"
append_summary "coarse_gate" "$STATUS_GATE" "lexical" "$GATE_TSV"

{
    echo "============================================================"
    echo " QMSum Coarse Segment Gate Compare Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
    echo ""
    echo "Baseline log: $BASE_LOG_ROOT"
    echo "Coarse gate log: $GATE_LOG_ROOT"
    echo "Summary TSV: $SUMMARY_TSV"
} | tee "$SUMMARY_TXT"

if [ "$STATUS_BASE" -ne 0 ] || [ "$STATUS_GATE" -ne 0 ]; then
    exit 1
fi
