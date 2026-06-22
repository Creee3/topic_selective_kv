#!/bin/bash
# ============================================================================
# Sweep lexical candidate-prefilter budget for the current QMSum mainline.
#
# Goal:
#   reduce online Q-K scoring cost while preserving answer quality under the
#   currently best detail-query budget setting.
#
# Fixed mainline defaults:
#   lexical coarse routing
#   -> top1 topic
#   -> topic-local Q-K chunk routing
#   -> dynamic_detail_top_k=12
#   -> qk_then_time
#   -> turns + strict
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_prefilter_budget_sweep.sh
#
# Optional overrides:
#   GPU_ID=3 START_DOC=5 END_DOC=10 MAX_QUERIES=5 \
#   PREFILTER_CASES="4:24:64 4:24:96 6:48:128" \
#   bash scripts_qmsum/run_qmsum_prefilter_budget_sweep.sh
#
# Case format:
#   factor:min_keep:max_keep
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID=${GPU_ID:-0}
NUM_NODES=${NUM_NODES:-4}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-10}
MAX_QUERIES=${MAX_QUERIES:-5}
ROUTE_TOP_K=${ROUTE_TOP_K:-12}
DYNAMIC_SUMMARY_TOP_K=${DYNAMIC_SUMMARY_TOP_K:-16}
DYNAMIC_DETAIL_TOP_K=${DYNAMIC_DETAIL_TOP_K:-12}
DYNAMIC_BALANCED_TOP_K=${DYNAMIC_BALANCED_TOP_K:-12}
ANSWER_EVIDENCE_ORDER=${ANSWER_EVIDENCE_ORDER:-qk_then_time}
SELECTED_ANSWER_CONTEXT_MODE=${SELECTED_ANSWER_CONTEXT_MODE:-turns}
ANSWER_PROMPT_STYLE=${ANSWER_PROMPT_STYLE:-strict}
ANSWER_EVIDENCE_MAX_ENTRIES=${ANSWER_EVIDENCE_MAX_ENTRIES:-80}
ANSWER_EVIDENCE_MAX_CHARS=${ANSWER_EVIDENCE_MAX_CHARS:-600}
ROUTE_CANDIDATE_PREFILTER=${ROUTE_CANDIDATE_PREFILTER:-lexical}
ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=${ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO:-0.0}
PREFILTER_CASES_STR=${PREFILTER_CASES:-"4:24:64 4:24:96 4:48:96 6:24:96 6:48:96 6:48:128"}
CASE_SUMMARY_TAG_SUFFIX=${CASE_SUMMARY_TAG_SUFFIX:-}
FETCH_BANDWIDTH_GBPS=${FETCH_BANDWIDTH_GBPS:-25.0}
PER_NODE_RTT_MS=${PER_NODE_RTT_MS:-1.0}
PER_SEGMENT_OVERHEAD_MS=${PER_SEGMENT_OVERHEAD_MS:-0.15}
DECODE_STARTUP_MS=${DECODE_STARTUP_MS:-15.0}
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}
LOG_ROOT=${LOG_ROOT:-logs/qmsum_prefilter_budget_sweep}
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

read -r -a PREFILTER_CASES <<< "$PREFILTER_CASES_STR"

mkdir -p "$LOG_ROOT"

cat > "$SUMMARY_TSV" <<EOF
case	status	prefilter_mode	min_prune_ratio	factor	min_keep	max_keep	avg_full_f1	avg_sel_f1	avg_delta	detail_full_f1	detail_sel_f1	detail_delta	detail_recall	detail_precision	detail_ctx_save_pct	avg_candidates_before	avg_prefilter_pool	avg_candidates_after	avg_prefilter_ms	avg_qk_ms	avg_routing_ms	avg_selected_ttft_ms	avg_selected_fetch_ms	overall_top1_hit_pct	detail_top1_hit_pct
EOF

has_complete_log() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 1
    fi
    grep -Fq "QMSum routing summary" "$file" || return 1
    grep -Fq "Answer quality (full vs selective):" "$file" || return 1
    grep -Fq "Saved to outputs/" "$file" || return 1
    return 0
}

append_failure() {
    local case_name="$1"
    local status="$2"
    local mode="$3"
    local factor="$4"
    local min_keep="$5"
    local max_keep="$6"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$case_name" "$status" "$mode" "$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO" "$factor" "$min_keep" "$max_keep" \
        "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" \
        "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" \
        >> "$SUMMARY_TSV"
}

append_summary_from_tsv() {
    local case_name="$1"
    local mode="$2"
    local factor="$3"
    local min_keep="$4"
    local max_keep="$5"
    local case_tsv="$6"

    if [ ! -f "$case_tsv" ]; then
        append_failure "$case_name" "MISSING_TSV" "$mode" "$factor" "$min_keep" "$max_keep"
        return
    fi

    CASE_NAME="$case_name" \
    PREFILTER_MODE="$mode" \
    MIN_PRUNE_RATIO="$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO" \
    FACTOR="$factor" \
    MIN_KEEP="$min_keep" \
    MAX_KEEP="$max_keep" \
    CASE_TSV="$case_tsv" \
    python - <<'PY' >> "$SUMMARY_TSV"
import csv
import os

case_name = os.environ["CASE_NAME"]
mode = os.environ["PREFILTER_MODE"]
min_prune_ratio = os.environ["MIN_PRUNE_RATIO"]
factor = os.environ["FACTOR"]
min_keep = os.environ["MIN_KEEP"]
max_keep = os.environ["MAX_KEEP"]
case_tsv = os.environ["CASE_TSV"]

with open(case_tsv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

if not rows:
    print("\t".join([case_name, "EMPTY_TSV", mode, min_prune_ratio, factor, min_keep, max_keep] + ["FAILED"] * 19))
    raise SystemExit

def avg(subset, field):
    return sum(float(r[field]) for r in subset) / len(subset) if subset else 0.0

def pct(subset, field):
    return 100.0 * sum(1 for r in subset if r[field] == "1") / len(subset) if subset else 0.0

detail = [r for r in rows if r["query_budget_type"] == "detail"]
summary = [r for r in rows if r["query_budget_type"] == "summary"]

values = [
    case_name,
    "OK",
    mode,
    min_prune_ratio,
    factor,
    min_keep,
    max_keep,
    f"{avg(rows, 'full_answer_f1'):.4f}",
    f"{avg(rows, 'selected_answer_f1'):.4f}",
    f"{avg(rows, 'answer_f1_delta'):.4f}",
    f"{avg(detail, 'full_answer_f1'):.4f}",
    f"{avg(detail, 'selected_answer_f1'):.4f}",
    f"{avg(detail, 'answer_f1_delta'):.4f}",
    f"{avg(detail, 'selected_turn_recall'):.4f}",
    f"{avg(detail, 'selected_turn_precision'):.4f}",
    f"{avg(detail, 'ctx_token_saving_pct'):.1f}",
    f"{avg(rows, 'num_candidates_before_prefilter'):.1f}",
    f"{avg(rows, 'candidate_prefilter_pool_size'):.1f}",
    f"{avg(rows, 'num_candidates_after_prefilter'):.1f}",
    f"{avg(rows, 'candidate_prefilter_ms'):.2f}",
    f"{avg(rows, 'qk_scoring_ms'):.2f}",
    f"{avg(rows, 'routing_overhead_ms'):.2f}",
    f"{avg(rows, 'selected_ttft_ms'):.2f}",
    f"{avg(rows, 'selected_fetch_latency_ms'):.2f}",
    f"{pct(rows, 'lexical_top1_hit'):.1f}",
    f"{pct(detail, 'lexical_top1_hit'):.1f}",
]
print("\t".join(values))
PY
}

run_case() {
    local spec="$1"
    local factor min_keep max_keep
    IFS=":" read -r factor min_keep max_keep <<< "$spec"

    if [ -z "$factor" ] || [ -z "$min_keep" ] || [ -z "$max_keep" ]; then
        echo "Skip malformed case spec: $spec"
        append_failure "bad_spec_${spec}" "BAD_SPEC" "$ROUTE_CANDIDATE_PREFILTER" \
            "${factor:-NA}" "${min_keep:-NA}" "${max_keep:-NA}"
        return
    fi

    local case_name="prefilter_f${factor}_m${min_keep}_x${max_keep}${CASE_SUMMARY_TAG_SUFFIX}"
    local case_tag="${case_name}_detail${DYNAMIC_DETAIL_TOP_K}_qk_then_time_turns_strict"
    local log_dir="$LOG_ROOT/$case_name"
    local log_file="$log_dir/mainline_answer_eval.log"
    local case_tsv="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${case_tag}.tsv"

    mkdir -p "$log_dir"

    echo ""
    echo "============================================================"
    echo " Running case: $case_name"
    echo " prefilter_mode=$ROUTE_CANDIDATE_PREFILTER"
    echo " factor=$factor min_keep=$min_keep max_keep=$max_keep"
    echo " docs=$START_DOC:$END_DOC"
    echo " max_queries=$MAX_QUERIES"
    echo " log_dir=$log_dir"
    echo "============================================================"

    if [ "$RESUME_IF_LOG_OK" -eq 1 ] && has_complete_log "$log_file"; then
        echo " Existing complete log detected. Reuse: $log_file"
        append_summary_from_tsv "$case_name" "$ROUTE_CANDIDATE_PREFILTER" \
            "$factor" "$min_keep" "$max_keep" "$case_tsv"
        return
    fi

    GPU_ID="$GPU_ID" \
    NUM_NODES="$NUM_NODES" \
    START_DOC="$START_DOC" \
    END_DOC="$END_DOC" \
    MAX_QUERIES="$MAX_QUERIES" \
    ROUTE_TOP_K="$ROUTE_TOP_K" \
    DYNAMIC_ROUTE_BUDGET=1 \
    DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K" \
    DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K" \
    DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K" \
    ANSWER_EVIDENCE_ORDER="$ANSWER_EVIDENCE_ORDER" \
    SELECTED_ANSWER_CONTEXT_MODE="$SELECTED_ANSWER_CONTEXT_MODE" \
    ANSWER_PROMPT_STYLE="$ANSWER_PROMPT_STYLE" \
    ANSWER_EVIDENCE_MAX_ENTRIES="$ANSWER_EVIDENCE_MAX_ENTRIES" \
    ANSWER_EVIDENCE_MAX_CHARS="$ANSWER_EVIDENCE_MAX_CHARS" \
    ROUTE_CANDIDATE_PREFILTER="$ROUTE_CANDIDATE_PREFILTER" \
    ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO="$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO" \
    ROUTE_CANDIDATE_PREFILTER_FACTOR="$factor" \
    ROUTE_CANDIDATE_PREFILTER_MIN_KEEP="$min_keep" \
    ROUTE_CANDIDATE_PREFILTER_MAX_KEEP="$max_keep" \
    FETCH_BANDWIDTH_GBPS="$FETCH_BANDWIDTH_GBPS" \
    PER_NODE_RTT_MS="$PER_NODE_RTT_MS" \
    PER_SEGMENT_OVERHEAD_MS="$PER_SEGMENT_OVERHEAD_MS" \
    DECODE_STARTUP_MS="$DECODE_STARTUP_MS" \
    CASE_SUMMARY_TAG="$case_tag" \
    LOG_DIR="$log_dir" \
    RESUME_IF_LOG_OK=0 \
    bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh"
    local status=$?

    if [ "$status" -ne 0 ]; then
        append_failure "$case_name" "RUN_FAIL" "$ROUTE_CANDIDATE_PREFILTER" \
            "$factor" "$min_keep" "$max_keep"
        return
    fi

    append_summary_from_tsv "$case_name" "$ROUTE_CANDIDATE_PREFILTER" \
        "$factor" "$min_keep" "$max_keep" "$case_tsv"
}

echo "============================================================"
echo " QMSum Prefilter-Budget Sweep"
echo " GPU_ID=$GPU_ID"
echo " docs=$START_DOC:$END_DOC"
echo " MAX_QUERIES=$MAX_QUERIES"
echo " ROUTE_TOP_K=$ROUTE_TOP_K"
echo " DYNAMIC_DETAIL_TOP_K=$DYNAMIC_DETAIL_TOP_K"
echo " ROUTE_CANDIDATE_PREFILTER=$ROUTE_CANDIDATE_PREFILTER"
echo " ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO"
echo " CASE_SUMMARY_TAG_SUFFIX=$CASE_SUMMARY_TAG_SUFFIX"
echo " PREFILTER_CASES=${PREFILTER_CASES[*]}"
echo " FETCH_BANDWIDTH_GBPS=$FETCH_BANDWIDTH_GBPS"
echo " PER_NODE_RTT_MS=$PER_NODE_RTT_MS"
echo " PER_SEGMENT_OVERHEAD_MS=$PER_SEGMENT_OVERHEAD_MS"
echo " DECODE_STARTUP_MS=$DECODE_STARTUP_MS"
echo "============================================================"

for spec in "${PREFILTER_CASES[@]}"; do
    run_case "$spec"
done

{
    echo "============================================================"
    echo " QMSum Prefilter-Budget Sweep Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Sweep complete"
echo " Summary: $SUMMARY_TXT"
echo "============================================================"
