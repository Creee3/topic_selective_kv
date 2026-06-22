#!/bin/bash
# ============================================================================
# Compare current top-1 routing vs local neighbor expansion around selected chunks.
#
# This keeps topic routing unchanged and only tests whether adding adjacent
# chunks improves evidence recall / answer quality without a large TTFT cost.
#
# Status:
#   The 0:30, max_queries=5 run showed no quality gain for neighbor_expand=1:
#   selected F1 and turn metrics were identical, while selected KV and TTFT
#   increased. Keep route_neighbor_expand=0 for the current mainline; use this
#   script only as a historical/explicit ablation.
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   START_DOC=0 END_DOC=30 MAX_QUERIES=5 GPU_A=0 GPU_B=1 \
#   bash scripts_qmsum/run_qmsum_neighbor_rescue_compare_dual_gpu.sh
# ============================================================================

set -u

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [ "$SCRIPT_DIR" = "$SCRIPT_PATH" ]; then
    SCRIPT_DIR="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"

GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}

START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-30}
MAX_QUERIES=${MAX_QUERIES:-5}
NUM_NODES=${NUM_NODES:-4}

BASE_NEIGHBOR_EXPAND=${BASE_NEIGHBOR_EXPAND:-0}
RESCUE_NEIGHBOR_EXPAND=${RESCUE_NEIGHBOR_EXPAND:-1}

ROUTE_TOP_K=${ROUTE_TOP_K:-12}
QK_SCORE_BATCH_SIZE=${QK_SCORE_BATCH_SIZE:-64}
CACHE_CANDIDATE_KEYS=${CACHE_CANDIDATE_KEYS:-1}

DYNAMIC_SUMMARY_TOP_K=${DYNAMIC_SUMMARY_TOP_K:-16}
DYNAMIC_DETAIL_TOP_K=${DYNAMIC_DETAIL_TOP_K:-12}
DYNAMIC_BALANCED_TOP_K=${DYNAMIC_BALANCED_TOP_K:-12}
DYNAMIC_CANDIDATE_POOL_BUDGET_MAP=${DYNAMIC_CANDIDATE_POOL_BUDGET_MAP:-summary:96,detail:48,balanced:56,default:56}
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
ANSWER_PROMPT_STYLE=${ANSWER_PROMPT_STYLE:-grounded}
ANSWER_MAX_NEW_TOKENS=${ANSWER_MAX_NEW_TOKENS:-96}
ANSWER_EVIDENCE_MAX_ENTRIES=${ANSWER_EVIDENCE_MAX_ENTRIES:-80}
ANSWER_EVIDENCE_MAX_CHARS=${ANSWER_EVIDENCE_MAX_CHARS:-600}

FETCH_BANDWIDTH_GBPS=${FETCH_BANDWIDTH_GBPS:-25.0}
PER_NODE_RTT_MS=${PER_NODE_RTT_MS:-1.0}
PER_SEGMENT_OVERHEAD_MS=${PER_SEGMENT_OVERHEAD_MS:-0.15}
DECODE_STARTUP_MS=${DECODE_STARTUP_MS:-15.0}
QUERY_TOKENIZER_WARMUP=${QUERY_TOKENIZER_WARMUP:-1}

LOG_ROOT=${LOG_ROOT:-logs/qmsum_neighbor_rescue_compare_${START_DOC}_${END_DOC}_q${MAX_QUERIES}}
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

case_tag_for_neighbor() {
    local neighbor_expand="$1"
    echo "neighborrescue_n${neighbor_expand}_dyn_grounded96_s${ROUTE_COARSE_SEGMENT_SIZE}_r${ROUTE_COARSE_SEGMENT_KEEP_RATIO}_m${ROUTE_COARSE_SEGMENT_MIN_KEEP}_k${ROUTE_TOP_K}_batch${QK_SCORE_BATCH_SIZE}_q${MAX_QUERIES}"
}

run_case() {
    local gpu_id="$1"
    local neighbor_expand="$2"
    local tag
    tag="$(case_tag_for_neighbor "$neighbor_expand")"
    local log_dir="$LOG_ROOT/neighbor${neighbor_expand}"

    GPU_ID="$gpu_id" \
    START_DOC="$START_DOC" \
    END_DOC="$END_DOC" \
    MAX_QUERIES="$MAX_QUERIES" \
    NUM_NODES="$NUM_NODES" \
    HIER_TOP_TOPICS=1 \
    ROUTE_TOP_K="$ROUTE_TOP_K" \
    ROUTE_NEIGHBOR_EXPAND="$neighbor_expand" \
    QK_SCORE_BATCH_SIZE="$QK_SCORE_BATCH_SIZE" \
    CACHE_CANDIDATE_KEYS="$CACHE_CANDIDATE_KEYS" \
    DYNAMIC_ROUTE_BUDGET=1 \
    DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
    DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
    DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
    DYNAMIC_CANDIDATE_POOL_BUDGET=1 \
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
    ANSWER_PROMPT_STYLE="$ANSWER_PROMPT_STYLE" \
    ANSWER_MAX_NEW_TOKENS="$ANSWER_MAX_NEW_TOKENS" \
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
    local label="$1"
    local neighbor_expand="$2"
    local status="$3"
    local tag
    tag="$(case_tag_for_neighbor "$neighbor_expand")"
    local case_tsv="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${tag}.tsv"

    if [ "$status" -ne 0 ] || [ ! -f "$case_tsv" ]; then
        {
            printf "%s\tRUN_FAIL\t%s" "$label" "$neighbor_expand"
            for _field in \
                n avg_full_f1 avg_selected_f1 avg_oracle_f1 \
                avg_selected_minus_full avg_selected_minus_oracle \
                selected_bad_output_rate avg_turn_recall avg_turn_precision avg_turn_f1 \
                zero_recall_rate selected_zero_f1_rate avg_selected_chunks \
                avg_selected_kv_mib avg_qk_candidates avg_qk_ms avg_qk_total_ms \
                avg_selected_ttft_ms ctx_token_saving_pct
            do
                printf "\tFAILED"
            done
            printf "\t%s\n" "$case_tsv"
        } >> "$SUMMARY_TSV"
        return
    fi

    LABEL="$label" NEIGHBOR="$neighbor_expand" CASE_TSV="$case_tsv" python - <<'PY' >> "$SUMMARY_TSV"
import csv
import os

label = os.environ["LABEL"]
neighbor = os.environ["NEIGHBOR"]
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

zero_recall = sum(1 for row in rows if float(row.get("selected_turn_recall") or 0.0) == 0.0)
sel_zero_f1 = sum(1 for row in rows if float(row.get("selected_answer_f1") or 0.0) == 0.0)

values = [
    label,
    "OK",
    neighbor,
    str(len(rows)),
    f"{avg('full_answer_f1'):.4f}",
    f"{avg('selected_answer_f1'):.4f}",
    f"{avg('oracle_answer_f1'):.4f}",
    f"{avg('answer_f1_delta'):.4f}",
    f"{avg('selected_answer_f1_delta_vs_oracle'):.4f}",
    f"{rate('selected_bad_output'):.4f}",
    f"{avg('selected_turn_recall'):.4f}",
    f"{avg('selected_turn_precision'):.4f}",
    f"{avg('selected_turn_f1'):.4f}",
    f"{zero_recall / len(rows):.4f}" if rows else "0.0000",
    f"{sel_zero_f1 / len(rows):.4f}" if rows else "0.0000",
    f"{avg('transfer_virtual_node_segments'):.1f}",
    f"{avg('selected_kv_mib'):.2f}",
    f"{avg('num_candidates_scored_qk'):.1f}",
    f"{avg('qk_scoring_ms'):.2f}",
    f"{avg('qk_total_stage_ms'):.2f}",
    f"{avg('selected_ttft_ms'):.2f}",
    f"{avg('ctx_token_saving_pct'):.1f}",
    case_tsv,
]
print("\t".join(values))
PY
}

echo "============================================================"
echo " QMSum Neighbor Rescue Compare Dual GPU"
echo " docs=$START_DOC:$END_DOC MAX_QUERIES=$MAX_QUERIES"
echo " GPU_A=$GPU_A -> neighbor_expand=$BASE_NEIGHBOR_EXPAND"
echo " GPU_B=$GPU_B -> neighbor_expand=$RESCUE_NEIGHBOR_EXPAND"
echo " dynamic_pool_map=$DYNAMIC_CANDIDATE_POOL_BUDGET_MAP"
echo " answer_max_new_tokens=$ANSWER_MAX_NEW_TOKENS"
echo " LOG_ROOT=$LOG_ROOT"
echo "============================================================"

run_case "$GPU_A" "$BASE_NEIGHBOR_EXPAND"
PID_A=$!
run_case "$GPU_B" "$RESCUE_NEIGHBOR_EXPAND"
PID_B=$!

wait "$PID_A"
STATUS_A=$?
wait "$PID_B"
STATUS_B=$?

cat > "$SUMMARY_TSV" <<EOF
case	status	route_neighbor_expand	n	avg_full_f1	avg_selected_f1	avg_oracle_f1	avg_selected_minus_full	avg_selected_minus_oracle	selected_bad_output_rate	avg_turn_recall	avg_turn_precision	avg_turn_f1	zero_recall_rate	selected_zero_f1_rate	avg_selected_segments	avg_selected_kv_mib	avg_qk_candidates	avg_qk_ms	avg_qk_total_ms	avg_selected_ttft_ms	ctx_token_saving_pct	case_tsv
EOF
append_summary "neighbor0_current" "$BASE_NEIGHBOR_EXPAND" "$STATUS_A"
append_summary "neighbor${RESCUE_NEIGHBOR_EXPAND}_rescue" "$RESCUE_NEIGHBOR_EXPAND" "$STATUS_B"

{
    echo "============================================================"
    echo " QMSum Neighbor Rescue Compare Summary"
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
