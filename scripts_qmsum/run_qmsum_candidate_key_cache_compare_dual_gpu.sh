#!/bin/bash
# ============================================================================
# Compare candidate-key cache OFF vs ON on two GPUs in parallel.
#
# Goal:
#   test whether reusing stacked candidate K tensors across queries reduces
#   online routing cost without changing answer quality.
#
# Default setup:
#   GPU 3 -> CACHE_CANDIDATE_KEYS=0
#   GPU 4 -> CACHE_CANDIDATE_KEYS=1
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_candidate_key_cache_compare_dual_gpu.sh
#
# Optional overrides:
#   GPU_A=3 GPU_B=4 START_DOC=5 END_DOC=10 MAX_QUERIES=5 \
#   bash scripts_qmsum/run_qmsum_candidate_key_cache_compare_dual_gpu.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_A=${GPU_A:-3}
GPU_B=${GPU_B:-4}
CACHE_A=${CACHE_A:-0}
CACHE_B=${CACHE_B:-1}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-10}
MAX_QUERIES=${MAX_QUERIES:-5}
NUM_NODES=${NUM_NODES:-4}
ROUTE_TOP_K=${ROUTE_TOP_K:-12}
QK_SCORE_BATCH_SIZE=${QK_SCORE_BATCH_SIZE:-64}
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
QUERY_TOKENIZER_WARMUP=${QUERY_TOKENIZER_WARMUP:-1}
LOG_ROOT=${LOG_ROOT:-logs/qmsum_candidate_key_cache_compare_dual_gpu}
LOG_DIR_A="$LOG_ROOT/cache_${CACHE_A}"
LOG_DIR_B="$LOG_ROOT/cache_${CACHE_B}"
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

echo "============================================================"
echo " QMSum Candidate-Key Cache Compare Dual GPU"
echo " GPU_A=$GPU_A cache=$CACHE_A"
echo " GPU_B=$GPU_B cache=$CACHE_B"
echo " docs=$START_DOC:$END_DOC"
echo " MAX_QUERIES=$MAX_QUERIES"
echo " ROUTE_TOP_K=$ROUTE_TOP_K"
echo " QK_SCORE_BATCH_SIZE=$QK_SCORE_BATCH_SIZE"
echo " QUERY_TOKENIZER_WARMUP=$QUERY_TOKENIZER_WARMUP"
echo " DYNAMIC_DETAIL_TOP_K=$DYNAMIC_DETAIL_TOP_K"
echo " PREFILTER=${ROUTE_CANDIDATE_PREFILTER} factor=$ROUTE_CANDIDATE_PREFILTER_FACTOR min=$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP max=$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP"
echo "============================================================"

GPU_ID="$GPU_A" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
NUM_NODES="$NUM_NODES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
QK_SCORE_BATCH_SIZE="$QK_SCORE_BATCH_SIZE" \
DYNAMIC_ROUTE_BUDGET=1 \
DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
ROUTE_CANDIDATE_PREFILTER_FACTOR="$ROUTE_CANDIDATE_PREFILTER_FACTOR" \
ROUTE_CANDIDATE_PREFILTER_MIN_KEEP="$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP" \
ROUTE_CANDIDATE_PREFILTER_MAX_KEEP="$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP" \
CACHE_CANDIDATE_KEYS="$CACHE_A" \
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
CASE_SUMMARY_TAG="keycache_${CACHE_A}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}" \
LOG_DIR="$LOG_DIR_A" \
bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh" &
PID_A=$!

GPU_ID="$GPU_B" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
NUM_NODES="$NUM_NODES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
QK_SCORE_BATCH_SIZE="$QK_SCORE_BATCH_SIZE" \
DYNAMIC_ROUTE_BUDGET=1 \
DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
ROUTE_CANDIDATE_PREFILTER_FACTOR="$ROUTE_CANDIDATE_PREFILTER_FACTOR" \
ROUTE_CANDIDATE_PREFILTER_MIN_KEEP="$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP" \
ROUTE_CANDIDATE_PREFILTER_MAX_KEEP="$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP" \
CACHE_CANDIDATE_KEYS="$CACHE_B" \
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
CASE_SUMMARY_TAG="keycache_${CACHE_B}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}" \
LOG_DIR="$LOG_DIR_B" \
bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh" &
PID_B=$!

wait $PID_A
STATUS_A=$?

wait $PID_B
STATUS_B=$?

cat > "$SUMMARY_TSV" <<EOF
case	status	cache_candidate_keys	qk_score_batch_size	avg_full_f1	avg_sel_f1	avg_delta	avg_key_prepare_ms	avg_qk_scoring_ms	avg_routing_ms	avg_routing_ms_steady	avg_selected_ttft_ms	avg_selected_ttft_ms_steady	ctx_token_saving_pct
EOF

append_summary_from_tsv() {
    local case_name="$1"
    local status="$2"
    local cache_flag="$3"
    local case_tsv="$4"

    if [ "$status" -ne 0 ] || [ ! -f "$case_tsv" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "FAILED" "$cache_flag" "$QK_SCORE_BATCH_SIZE" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" \
            >> "$SUMMARY_TSV"
        return
    fi

    CASE_NAME="$case_name" CACHE_FLAG="$cache_flag" CASE_TSV="$case_tsv" python - <<'PY' >> "$SUMMARY_TSV"
import csv
import os

case_name = os.environ["CASE_NAME"]
cache_flag = os.environ["CACHE_FLAG"]
case_tsv = os.environ["CASE_TSV"]
batch_size = os.environ.get("QK_SCORE_BATCH_SIZE", "")

with open(case_tsv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

if not rows:
    print("\t".join([case_name, "EMPTY_TSV", cache_flag, batch_size] + ["FAILED"] * 10))
    raise SystemExit

def avg(field):
    return sum(float(r[field]) for r in rows) / len(rows) if rows else 0.0

steady_rows = [r for r in rows if str(r.get("timing_is_first_query", "0")) != "1"]

def avg_steady(field):
    return sum(float(r[field]) for r in steady_rows) / len(steady_rows) if steady_rows else 0.0

values = [
    case_name,
    "OK",
    cache_flag,
    batch_size,
    f"{avg('full_answer_f1'):.4f}",
    f"{avg('selected_answer_f1'):.4f}",
    f"{avg('answer_f1_delta'):.4f}",
    f"{avg('candidate_key_prepare_ms'):.2f}",
    f"{avg('qk_scoring_ms'):.2f}",
    f"{avg('routing_overhead_ms'):.2f}",
    f"{avg_steady('routing_overhead_ms'):.2f}",
    f"{avg('selected_ttft_ms'):.2f}",
    f"{avg_steady('selected_ttft_ms'):.2f}",
    f"{avg('ctx_token_saving_pct'):.1f}",
]
print("\t".join(values))
PY
}

TSV_A="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_keycache_${CACHE_A}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}.tsv"
TSV_B="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_keycache_${CACHE_B}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}.tsv"

append_summary_from_tsv "cache_${CACHE_A}" "$STATUS_A" "$CACHE_A" "$TSV_A"
append_summary_from_tsv "cache_${CACHE_B}" "$STATUS_B" "$CACHE_B" "$TSV_B"

{
    echo "============================================================"
    echo " QMSum Candidate-Key Cache Compare Summary"
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
