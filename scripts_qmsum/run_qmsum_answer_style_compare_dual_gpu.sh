#!/bin/bash
# ============================================================================
# Compare answer prompt styles while keeping routing fixed.
#
# Default routing:
#   lexical topic routing
#   qk_then_time selected evidence
#   coarse segment gate s4 r0.65 m64
#
# Default comparison:
#   GPU_A -> ANSWER_PROMPT_STYLE=strict
#   GPU_B -> ANSWER_PROMPT_STYLE=grounded
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   START_DOC=5 END_DOC=20 MAX_QUERIES=5 \
#   GPU_A=0 GPU_B=1 \
#   bash scripts_qmsum/run_qmsum_answer_style_compare_dual_gpu.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
STYLE_A=${STYLE_A:-strict}
STYLE_B=${STYLE_B:-grounded}

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
DYNAMIC_CANDIDATE_POOL_BUDGET=${DYNAMIC_CANDIDATE_POOL_BUDGET:-0}
DYNAMIC_CANDIDATE_POOL_BUDGET_MAP=${DYNAMIC_CANDIDATE_POOL_BUDGET_MAP:-summary:64,detail:48,balanced:48,default:48}
DYNAMIC_CANDIDATE_POOL_MIN_KEEP=${DYNAMIC_CANDIDATE_POOL_MIN_KEEP:-24}

ROUTE_CANDIDATE_PREFILTER=${ROUTE_CANDIDATE_PREFILTER:-lexical}
ROUTE_CANDIDATE_PREFILTER_FACTOR=${ROUTE_CANDIDATE_PREFILTER_FACTOR:-6}
ROUTE_CANDIDATE_PREFILTER_MIN_KEEP=${ROUTE_CANDIDATE_PREFILTER_MIN_KEEP:-48}
ROUTE_CANDIDATE_PREFILTER_MAX_KEEP=${ROUTE_CANDIDATE_PREFILTER_MAX_KEEP:-128}
ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=${ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO:-0.0}

ROUTE_COARSE_SEGMENT_GATE=${ROUTE_COARSE_SEGMENT_GATE:-lexical}
ROUTE_COARSE_SEGMENT_SIZE=${ROUTE_COARSE_SEGMENT_SIZE:-4}
ROUTE_COARSE_SEGMENT_KEEP_RATIO=${ROUTE_COARSE_SEGMENT_KEEP_RATIO:-0.65}
ROUTE_COARSE_SEGMENT_MIN_KEEP=${ROUTE_COARSE_SEGMENT_MIN_KEEP:-64}
ROUTE_COARSE_SEGMENT_MAX_KEEP=${ROUTE_COARSE_SEGMENT_MAX_KEEP:-0}

ANSWER_EVIDENCE_ORDER=${ANSWER_EVIDENCE_ORDER:-qk_then_time}
SELECTED_ANSWER_CONTEXT_MODE=${SELECTED_ANSWER_CONTEXT_MODE:-turns}
ANSWER_EVIDENCE_MAX_ENTRIES=${ANSWER_EVIDENCE_MAX_ENTRIES:-80}
ANSWER_EVIDENCE_MAX_CHARS=${ANSWER_EVIDENCE_MAX_CHARS:-600}

FETCH_BANDWIDTH_GBPS=${FETCH_BANDWIDTH_GBPS:-25.0}
PER_NODE_RTT_MS=${PER_NODE_RTT_MS:-1.0}
PER_SEGMENT_OVERHEAD_MS=${PER_SEGMENT_OVERHEAD_MS:-0.15}
DECODE_STARTUP_MS=${DECODE_STARTUP_MS:-15.0}
QUERY_TOKENIZER_WARMUP=${QUERY_TOKENIZER_WARMUP:-1}

LOG_ROOT=${LOG_ROOT:-logs/qmsum_answer_style_compare_dual_gpu}
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

run_case() {
    local gpu_id="$1"
    local style="$2"
    local case_name="style_${style}"
    local tag="answerstyle_${style}_coarsegate_s${ROUTE_COARSE_SEGMENT_SIZE}_r${ROUTE_COARSE_SEGMENT_KEEP_RATIO}_m${ROUTE_COARSE_SEGMENT_MIN_KEEP}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}"
    local log_dir="$LOG_ROOT/$case_name"

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
    DYNAMIC_CANDIDATE_POOL_BUDGET="$DYNAMIC_CANDIDATE_POOL_BUDGET" \
    DYNAMIC_CANDIDATE_POOL_BUDGET_MAP="$DYNAMIC_CANDIDATE_POOL_BUDGET_MAP" \
    DYNAMIC_CANDIDATE_POOL_MIN_KEEP="$DYNAMIC_CANDIDATE_POOL_MIN_KEEP" \
    ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
    ROUTE_CANDIDATE_PREFILTER_FACTOR="$ROUTE_CANDIDATE_PREFILTER_FACTOR" \
    ROUTE_CANDIDATE_PREFILTER_MIN_KEEP="$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP" \
    ROUTE_CANDIDATE_PREFILTER_MAX_KEEP="$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP" \
    ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO="$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO" \
    ROUTE_COARSE_SEGMENT_GATE="$ROUTE_COARSE_SEGMENT_GATE" \
    ROUTE_COARSE_SEGMENT_SIZE="$ROUTE_COARSE_SEGMENT_SIZE" \
    ROUTE_COARSE_SEGMENT_KEEP_RATIO="$ROUTE_COARSE_SEGMENT_KEEP_RATIO" \
    ROUTE_COARSE_SEGMENT_MIN_KEEP="$ROUTE_COARSE_SEGMENT_MIN_KEEP" \
    ROUTE_COARSE_SEGMENT_MAX_KEEP="$ROUTE_COARSE_SEGMENT_MAX_KEEP" \
    ANSWER_EVIDENCE_ORDER="$ANSWER_EVIDENCE_ORDER" \
    SELECTED_ANSWER_CONTEXT_MODE="$SELECTED_ANSWER_CONTEXT_MODE" \
    ANSWER_PROMPT_STYLE="$style" \
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

append_summary() {
    local style="$1"
    local status="$2"
    local tag="answerstyle_${style}_coarsegate_s${ROUTE_COARSE_SEGMENT_SIZE}_r${ROUTE_COARSE_SEGMENT_KEEP_RATIO}_m${ROUTE_COARSE_SEGMENT_MIN_KEEP}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}"
    local case_tsv="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${tag}.tsv"

    if [ "$status" -ne 0 ] || [ ! -f "$case_tsv" ]; then
        echo "$style	RUN_FAIL	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED" >> "$SUMMARY_TSV"
        return
    fi

    STYLE="$style" CASE_TSV="$case_tsv" python - <<'PY' >> "$SUMMARY_TSV"
import csv
import os

style = os.environ["STYLE"]
case_tsv = os.environ["CASE_TSV"]
with open(case_tsv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

def avg(field):
    vals = []
    for row in rows:
        value = row.get(field, "")
        if value in ("", None):
            continue
        vals.append(float(value))
    return sum(vals) / len(vals) if vals else 0.0

def rate(field):
    return sum(1 for row in rows if str(row.get(field, "")).strip() == "1") / len(rows) if rows else 0.0

values = [
    style,
    "OK",
    f"{avg('full_answer_f1'):.4f}",
    f"{avg('selected_answer_f1'):.4f}",
    f"{avg('oracle_answer_f1'):.4f}",
    f"{avg('answer_f1_delta'):.4f}",
    f"{rate('full_bad_output'):.4f}",
    f"{rate('selected_bad_output'):.4f}",
    f"{rate('oracle_bad_output'):.4f}",
    f"{avg('selected_turn_recall'):.4f}",
    f"{avg('qk_scoring_ms'):.2f}",
    f"{avg('selected_ttft_ms'):.2f}",
    f"{avg('ctx_token_saving_pct'):.1f}",
]
print("\t".join(values))
PY
}

echo "============================================================"
echo " QMSum Answer Style Compare Dual GPU"
echo " docs=$START_DOC:$END_DOC MAX_QUERIES=$MAX_QUERIES"
echo " GPU_A=$GPU_A style=$STYLE_A"
echo " GPU_B=$GPU_B style=$STYLE_B"
echo " route_coarse_segment_gate=$ROUTE_COARSE_SEGMENT_GATE"
echo " gate size=$ROUTE_COARSE_SEGMENT_SIZE ratio=$ROUTE_COARSE_SEGMENT_KEEP_RATIO min=$ROUTE_COARSE_SEGMENT_MIN_KEEP"
echo " LOG_ROOT=$LOG_ROOT"
echo "============================================================"

run_case "$GPU_A" "$STYLE_A"
PID_A=$!
run_case "$GPU_B" "$STYLE_B"
PID_B=$!

wait $PID_A
STATUS_A=$?
wait $PID_B
STATUS_B=$?

cat > "$SUMMARY_TSV" <<EOF
style	status	avg_full_f1	avg_sel_f1	avg_oracle_f1	avg_delta	full_bad_output_rate	selected_bad_output_rate	oracle_bad_output_rate	avg_turn_recall	avg_qk_ms	avg_selected_ttft_ms	ctx_token_saving_pct
EOF
append_summary "$STYLE_A" "$STATUS_A"
append_summary "$STYLE_B" "$STATUS_B"

{
    echo "============================================================"
    echo " QMSum Answer Style Compare Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
    echo ""
    echo "Summary TSV: $SUMMARY_TSV"
    echo "Logs: $LOG_ROOT"
} | tee "$SUMMARY_TXT"

if [ "$STATUS_A" -ne 0 ] || [ "$STATUS_B" -ne 0 ]; then
    exit 1
fi
