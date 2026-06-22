#!/bin/bash
# ============================================================================
# 3-GPU sweep for Q-K scoring-layer cost/quality tradeoff.
#
# Cases:
#   GPU_A -> last layer only
#   GPU_B -> middle + last layers
#   GPU_C -> default 5-layer scoring
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_scoring_layers_3gpu_sweep.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-3}
GPU_C=${GPU_C:-4}

START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-10}
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

LOG_ROOT=${LOG_ROOT:-logs/qmsum_scoring_layers_3gpu_sweep}
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

run_case() {
    local gpu_id="$1"
    local case_name="$2"
    local scoring_layers="$3"
    local log_dir="$LOG_ROOT/$case_name"
    local tag="layers_${case_name}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}"

    GPU_ID="$gpu_id" \
    START_DOC="$START_DOC" \
    END_DOC="$END_DOC" \
    MAX_QUERIES="$MAX_QUERIES" \
    NUM_NODES="$NUM_NODES" \
    ROUTE_TOP_K="$ROUTE_TOP_K" \
    QK_SCORE_BATCH_SIZE="$QK_SCORE_BATCH_SIZE" \
    CACHE_CANDIDATE_KEYS="$CACHE_CANDIDATE_KEYS" \
    SCORING_LAYERS="$scoring_layers" \
    DYNAMIC_ROUTE_BUDGET=1 \
    DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
    DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
    DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
    ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
    ROUTE_CANDIDATE_PREFILTER_FACTOR="$ROUTE_CANDIDATE_PREFILTER_FACTOR" \
    ROUTE_CANDIDATE_PREFILTER_MIN_KEEP="$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP" \
    ROUTE_CANDIDATE_PREFILTER_MAX_KEEP="$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP" \
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
echo " QMSum Q-K Scoring-Layers 3-GPU Sweep"
echo " GPU_A=$GPU_A last_only      SCORING_LAYERS=-1"
echo " GPU_B=$GPU_B mid_last       SCORING_LAYERS=16,-1"
echo " GPU_C=$GPU_C default_5layer SCORING_LAYERS=default"
echo " docs=$START_DOC:$END_DOC  MAX_QUERIES=$MAX_QUERIES"
echo " cache_candidate_keys=$CACHE_CANDIDATE_KEYS  batch=$QK_SCORE_BATCH_SIZE"
echo "============================================================"

run_case "$GPU_A" "last_only" "-1"
PID_A=$!
run_case "$GPU_B" "mid_last" "16,-1"
PID_B=$!
run_case "$GPU_C" "default_5layer" ""
PID_C=$!

wait $PID_A
STATUS_A=$?
wait $PID_B
STATUS_B=$?
wait $PID_C
STATUS_C=$?

cat > "$SUMMARY_TSV" <<EOF
case	status	scoring_layers	avg_full_f1	avg_sel_f1	avg_oracle_f1	avg_delta	avg_sel_oracle_gap	avg_turn_recall	avg_turn_f1	avg_qk_model_ms	avg_qk_scoring_ms	avg_routing_ms	avg_routing_ms_steady	avg_selected_ttft_ms	avg_selected_ttft_ms_steady	ctx_token_saving_pct
EOF

append_summary_from_tsv() {
    local case_name="$1"
    local status="$2"
    local scoring_layers="$3"
    local case_tsv="$4"

    if [ "$status" -ne 0 ] || [ ! -f "$case_tsv" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "FAILED" "${scoring_layers:-default}" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" \
            >> "$SUMMARY_TSV"
        return
    fi

    CASE_NAME="$case_name" SCORING_LAYERS_LABEL="${scoring_layers:-default}" CASE_TSV="$case_tsv" python - <<'PY' >> "$SUMMARY_TSV"
import csv
import os

case_name = os.environ["CASE_NAME"]
scoring_layers = os.environ["SCORING_LAYERS_LABEL"]
case_tsv = os.environ["CASE_TSV"]

with open(case_tsv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

if not rows:
    print("\t".join([case_name, "EMPTY_TSV", scoring_layers] + ["FAILED"] * 14))
    raise SystemExit

def avg(field, source=None):
    items = rows if source is None else source
    return sum(float(r[field]) for r in items) / len(items) if items else 0.0

steady_rows = [r for r in rows if str(r.get("timing_is_first_query", "0")) != "1"]

values = [
    case_name,
    "OK",
    scoring_layers,
    f"{avg('full_answer_f1'):.4f}",
    f"{avg('selected_answer_f1'):.4f}",
    f"{avg('oracle_answer_f1'):.4f}",
    f"{avg('answer_f1_delta'):.4f}",
    f"{avg('selected_answer_f1_delta_vs_oracle'):.4f}",
    f"{avg('selected_turn_recall'):.4f}",
    f"{avg('selected_turn_f1'):.4f}",
    f"{avg('qk_model_inference_ms'):.2f}",
    f"{avg('qk_scoring_ms'):.2f}",
    f"{avg('routing_overhead_ms'):.2f}",
    f"{avg('routing_overhead_ms', steady_rows):.2f}",
    f"{avg('selected_ttft_ms'):.2f}",
    f"{avg('selected_ttft_ms', steady_rows):.2f}",
    f"{avg('ctx_token_saving_pct'):.1f}",
]
print("\t".join(values))
PY
}

TSV_A="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_layers_last_only_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}.tsv"
TSV_B="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_layers_mid_last_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}.tsv"
TSV_C="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_layers_default_5layer_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}.tsv"

append_summary_from_tsv "last_only" "$STATUS_A" "-1" "$TSV_A"
append_summary_from_tsv "mid_last" "$STATUS_B" "16,-1" "$TSV_B"
append_summary_from_tsv "default_5layer" "$STATUS_C" "default" "$TSV_C"

{
    echo "============================================================"
    echo " QMSum Q-K Scoring-Layers Sweep Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Scoring-layer sweep finished"
echo " last_only status=$STATUS_A"
echo " mid_last status=$STATUS_B"
echo " default_5layer status=$STATUS_C"
echo " Summary: $SUMMARY_TXT"
echo "============================================================"
