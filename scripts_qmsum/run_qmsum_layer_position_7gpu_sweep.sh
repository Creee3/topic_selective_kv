#!/bin/bash
# ============================================================================
# Multi-GPU sweep for Q-K scoring layer positions.
#
# This is the follow-up to the 3-GPU layer-count sweep.  The goal is to check
# whether early / middle / late layers carry different routing signal, instead
# of only asking "how many layers should we use?"
#
# Default layout uses the currently-idle GPUs from the user's node:
#   GPU 0 -> early_only       layers 0
#   GPU 1 -> last_only        layers -1
#   GPU 3 -> late_pair        layers 24,-1
#   GPU 4 -> mid_last         layers 16,-1
#   GPU 5 -> early_last       layers 0,-1
#   GPU 6 -> mid3_last        layers 8,16,-1
#   GPU 7 -> default_5layer   default layers 0,8,16,24,last
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_layer_position_7gpu_sweep.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_EARLY=${GPU_EARLY:-0}
GPU_LAST=${GPU_LAST:-1}
GPU_LATE_PAIR=${GPU_LATE_PAIR:-3}
GPU_MID_LAST=${GPU_MID_LAST:-4}
GPU_EARLY_LAST=${GPU_EARLY_LAST:-5}
GPU_MID3_LAST=${GPU_MID3_LAST:-6}
GPU_DEFAULT=${GPU_DEFAULT:-7}

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

LOG_ROOT=${LOG_ROOT:-logs/qmsum_layer_position_7gpu_sweep}
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

run_case() {
    local gpu_id="$1"
    local case_name="$2"
    local scoring_layers="$3"
    local log_dir="$LOG_ROOT/$case_name"
    local tag="layerpos_${case_name}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}"

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
echo " QMSum Q-K Layer-Position 7-GPU Sweep"
echo " docs=$START_DOC:$END_DOC  MAX_QUERIES=$MAX_QUERIES"
echo " cache_candidate_keys=$CACHE_CANDIDATE_KEYS  batch=$QK_SCORE_BATCH_SIZE"
echo " cases:"
echo "   GPU $GPU_EARLY      early_only      SCORING_LAYERS=0"
echo "   GPU $GPU_LAST       last_only       SCORING_LAYERS=-1"
echo "   GPU $GPU_LATE_PAIR  late_pair       SCORING_LAYERS=24,-1"
echo "   GPU $GPU_MID_LAST   mid_last        SCORING_LAYERS=16,-1"
echo "   GPU $GPU_EARLY_LAST early_last      SCORING_LAYERS=0,-1"
echo "   GPU $GPU_MID3_LAST  mid3_last       SCORING_LAYERS=8,16,-1"
echo "   GPU $GPU_DEFAULT    default_5layer  SCORING_LAYERS=default"
echo "============================================================"

CASE_NAMES=("early_only" "last_only" "late_pair" "mid_last" "early_last" "mid3_last" "default_5layer")
CASE_GPUS=("$GPU_EARLY" "$GPU_LAST" "$GPU_LATE_PAIR" "$GPU_MID_LAST" "$GPU_EARLY_LAST" "$GPU_MID3_LAST" "$GPU_DEFAULT")
CASE_LAYERS=("0" "-1" "24,-1" "16,-1" "0,-1" "8,16,-1" "")
CASE_STATUS=()
CASE_PIDS=()

for i in "${!CASE_NAMES[@]}"; do
    run_case "${CASE_GPUS[$i]}" "${CASE_NAMES[$i]}" "${CASE_LAYERS[$i]}"
    CASE_PIDS[$i]=$!
done

for i in "${!CASE_NAMES[@]}"; do
    wait "${CASE_PIDS[$i]}"
    CASE_STATUS[$i]=$?
done

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

for i in "${!CASE_NAMES[@]}"; do
    case_name="${CASE_NAMES[$i]}"
    case_layers="${CASE_LAYERS[$i]}"
    case_tsv="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_layerpos_${case_name}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}.tsv"
    append_summary_from_tsv "$case_name" "${CASE_STATUS[$i]}" "$case_layers" "$case_tsv"
done

{
    echo "============================================================"
    echo " QMSum Q-K Layer-Position Sweep Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Layer-position sweep finished"
for i in "${!CASE_NAMES[@]}"; do
    echo " ${CASE_NAMES[$i]} status=${CASE_STATUS[$i]}"
done
echo " Summary: $SUMMARY_TXT"
echo "============================================================"
