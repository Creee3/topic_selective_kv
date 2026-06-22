#!/bin/bash
# ============================================================================
# Sweep lexical coarse-segment gate budgets on up to four GPUs.
#
# Goal:
#   keep most of the Q-K / TTFT speedup from coarse gating while finding a
#   gentler budget that recovers selected-answer F1.
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   START_DOC=5 END_DOC=20 MAX_QUERIES=5 \
#   GPU_LIST="0 1 2 3" \
#   bash scripts_qmsum/run_qmsum_coarse_segment_gate_budget_sweep_4gpu.sh
#
# Optional overrides:
#   GATE_CASES="4:0.5:48:0 4:0.65:48:0 4:0.75:48:0 4:0.65:64:0 4:0.75:64:0"
#   RUN_BASELINE=1
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_LIST_STR=${GPU_LIST:-"0 1 2 3"}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-20}
MAX_QUERIES=${MAX_QUERIES:-5}
NUM_NODES=${NUM_NODES:-4}
RUN_BASELINE=${RUN_BASELINE:-1}

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

GATE_CASES_STR=${GATE_CASES:-"4:0.5:48:0 4:0.65:48:0 4:0.75:48:0 4:0.65:64:0 4:0.75:64:0"}

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
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}

LOG_ROOT=${LOG_ROOT:-logs/qmsum_coarse_segment_gate_budget_sweep_4gpu}
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

read -r -a GPUS <<< "$GPU_LIST_STR"
read -r -a GATE_CASES <<< "$GATE_CASES_STR"

mkdir -p "$LOG_ROOT"

if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "ERROR: GPU_LIST is empty"
    exit 1
fi

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

case_tag_for_spec() {
    local gate_mode="$1"
    local seg_size="$2"
    local keep_ratio="$3"
    local min_keep="$4"
    local max_keep="$5"

    if [ "$gate_mode" = "none" ]; then
        echo "coarsegate_base_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}"
    else
        local max_suffix=""
        if [ "$max_keep" != "0" ]; then
            max_suffix="_x${max_keep}"
        fi
        echo "coarsegate_lexical_s${seg_size}_r${keep_ratio}_m${min_keep}${max_suffix}_batch${QK_SCORE_BATCH_SIZE}_detail${DYNAMIC_DETAIL_TOP_K}_q${MAX_QUERIES}"
    fi
}

run_case() {
    local gpu_id="$1"
    local case_name="$2"
    local gate_mode="$3"
    local seg_size="$4"
    local keep_ratio="$5"
    local min_keep="$6"
    local max_keep="$7"

    local case_tag
    case_tag="$(case_tag_for_spec "$gate_mode" "$seg_size" "$keep_ratio" "$min_keep" "$max_keep")"
    local log_dir="$LOG_ROOT/$case_name"
    local log_file="$log_dir/mainline_answer_eval.log"

    mkdir -p "$log_dir"

    echo ""
    echo "============================================================"
    echo " Running case: $case_name"
    echo " GPU=$gpu_id gate=$gate_mode size=$seg_size keep_ratio=$keep_ratio min_keep=$min_keep max_keep=$max_keep"
    echo " tag=$case_tag"
    echo " log_dir=$log_dir"
    echo "============================================================"

    if [ "$RESUME_IF_LOG_OK" -eq 1 ] && has_complete_log "$log_file"; then
        echo " Existing complete log detected. Reuse: $log_file"
        return 0
    fi

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
    ROUTE_COARSE_SEGMENT_SIZE="$seg_size" \
    ROUTE_COARSE_SEGMENT_KEEP_RATIO="$keep_ratio" \
    ROUTE_COARSE_SEGMENT_MIN_KEEP="$min_keep" \
    ROUTE_COARSE_SEGMENT_MAX_KEEP="$max_keep" \
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
    CASE_SUMMARY_TAG="$case_tag" \
    LOG_DIR="$log_dir" \
    RESUME_IF_LOG_OK=0 \
    bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh"
}

append_summary() {
    local case_name="$1"
    local gate_mode="$2"
    local seg_size="$3"
    local keep_ratio="$4"
    local min_keep="$5"
    local max_keep="$6"

    local case_tag
    case_tag="$(case_tag_for_spec "$gate_mode" "$seg_size" "$keep_ratio" "$min_keep" "$max_keep")"
    local case_tsv="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${case_tag}.tsv"

    if [ ! -f "$case_tsv" ]; then
        echo "$case_name	MISSING_TSV	$gate_mode	$seg_size	$keep_ratio	$min_keep	$max_keep	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED	FAILED" >> "$SUMMARY_TSV"
        return
    fi

    CASE_NAME="$case_name" \
    GATE_MODE="$gate_mode" \
    SEG_SIZE="$seg_size" \
    KEEP_RATIO="$keep_ratio" \
    MIN_KEEP="$min_keep" \
    MAX_KEEP="$max_keep" \
    CASE_TSV="$case_tsv" \
    python - <<'PY' >> "$SUMMARY_TSV"
import csv
import os

case_name = os.environ["CASE_NAME"]
gate_mode = os.environ["GATE_MODE"]
seg_size = os.environ["SEG_SIZE"]
keep_ratio = os.environ["KEEP_RATIO"]
min_keep = os.environ["MIN_KEEP"]
max_keep = os.environ["MAX_KEEP"]
case_tsv = os.environ["CASE_TSV"]

with open(case_tsv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

if not rows:
    print("\t".join([case_name, "EMPTY_TSV", gate_mode, seg_size, keep_ratio, min_keep, max_keep] + ["FAILED"] * 23))
    raise SystemExit

steady_rows = [
    row for row in rows
    if str(row.get("timing_is_first_query", "")).strip() not in {"1", "true", "True"}
]
if not steady_rows:
    steady_rows = rows

def avg(source, field):
    values = []
    for row in source:
        value = row.get(field, "")
        if value in {"", None, "nan", "NaN"}:
            continue
        values.append(float(value))
    return sum(values) / len(values) if values else 0.0

values = [
    case_name,
    "OK",
    gate_mode,
    seg_size,
    keep_ratio,
    min_keep,
    max_keep,
    f"{avg(rows, 'full_answer_f1'):.4f}",
    f"{avg(rows, 'selected_answer_f1'):.4f}",
    f"{avg(rows, 'answer_f1_delta'):.4f}",
    f"{avg(rows, 'selected_turn_recall'):.4f}",
    f"{avg(rows, 'selected_turn_precision'):.4f}",
    f"{avg(rows, 'selected_turn_f1'):.4f}",
    f"{avg(rows, 'ctx_token_saving_pct'):.1f}",
    f"{avg(rows, 'num_candidates_before_prefilter'):.1f}",
    f"{avg(rows, 'num_candidates_after_prefilter'):.1f}",
    f"{avg(rows, 'coarse_segment_gate_before'):.1f}",
    f"{avg(rows, 'coarse_segment_gate_after'):.1f}",
    f"{100.0 * avg(rows, 'coarse_segment_gate_prune_ratio'):.1f}",
    f"{avg(rows, 'coarse_segment_gate_ms'):.2f}",
    f"{avg(rows, 'candidate_key_prepare_ms'):.2f}",
    f"{avg(rows, 'qk_model_inference_ms'):.2f}",
    f"{avg(rows, 'qk_score_aggregation_ms'):.2f}",
    f"{avg(rows, 'qk_scoring_ms'):.2f}",
    f"{avg(rows, 'routing_overhead_ms'):.2f}",
    f"{avg(rows, 'selected_ttft_ms'):.2f}",
    f"{avg(steady_rows, 'qk_scoring_ms'):.2f}",
    f"{avg(steady_rows, 'routing_overhead_ms'):.2f}",
    f"{avg(steady_rows, 'selected_ttft_ms'):.2f}",
    f"{avg(rows, 'full_ttft_ms'):.2f}",
]
print("\t".join(values))
PY
}

CASE_SPECS=()
if [ "$RUN_BASELINE" -eq 1 ]; then
    CASE_SPECS+=("baseline:none:0:0:0:0")
fi

for gate_case in "${GATE_CASES[@]}"; do
    IFS=':' read -r seg_size keep_ratio min_keep max_keep <<< "$gate_case"
    if [ -z "$seg_size" ] || [ -z "$keep_ratio" ] || [ -z "$min_keep" ] || [ -z "$max_keep" ]; then
        echo "ERROR: invalid gate case '$gate_case'. Expected segment_size:keep_ratio:min_keep:max_keep"
        exit 1
    fi
    case_name="gate_s${seg_size}_r${keep_ratio}_m${min_keep}"
    if [ "$max_keep" != "0" ]; then
        case_name="${case_name}_x${max_keep}"
    fi
    CASE_SPECS+=("${case_name}:lexical:${seg_size}:${keep_ratio}:${min_keep}:${max_keep}")
done

echo "============================================================"
echo " QMSum Coarse Segment Gate Budget Sweep 4GPU"
echo " docs=$START_DOC:$END_DOC  MAX_QUERIES=$MAX_QUERIES"
echo " GPUs=[$GPU_LIST_STR]"
echo " RUN_BASELINE=$RUN_BASELINE"
echo " GATE_CASES=[$GATE_CASES_STR]"
echo " QK_SCORE_BATCH_SIZE=$QK_SCORE_BATCH_SIZE"
echo " CACHE_CANDIDATE_KEYS=$CACHE_CANDIDATE_KEYS"
echo " LOG_ROOT=$LOG_ROOT"
echo "============================================================"

PIDS=()
WORKER_IDS=()

for gpu_idx in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$gpu_idx]}"
    (
        status=0
        for case_idx in "${!CASE_SPECS[@]}"; do
            if [ $((case_idx % ${#GPUS[@]})) -ne "$gpu_idx" ]; then
                continue
            fi
            IFS=':' read -r case_name gate_mode seg_size keep_ratio min_keep max_keep <<< "${CASE_SPECS[$case_idx]}"
            run_case "$gpu_id" "$case_name" "$gate_mode" "$seg_size" "$keep_ratio" "$min_keep" "$max_keep" || status=1
        done
        exit "$status"
    ) &
    PIDS+=("$!")
    WORKER_IDS+=("$gpu_id")
done

OVERALL_STATUS=0
for pid_idx in "${!PIDS[@]}"; do
    pid="${PIDS[$pid_idx]}"
    gpu_id="${WORKER_IDS[$pid_idx]}"
    if ! wait "$pid"; then
        echo "Worker on GPU $gpu_id failed"
        OVERALL_STATUS=1
    fi
done

cat > "$SUMMARY_TSV" <<EOF
case	status	gate_mode	segment_size	keep_ratio	min_keep	max_keep	avg_full_f1	avg_sel_f1	avg_delta	avg_turn_recall	avg_turn_precision	avg_turn_f1	ctx_token_saving_pct	avg_candidates_before	avg_candidates_after	avg_gate_before	avg_gate_after	avg_gate_prune_pct	avg_gate_ms	avg_candidate_key_prepare_ms	avg_qk_model_inference_ms	avg_qk_score_aggregation_ms	avg_qk_ms	avg_routing_ms	avg_selected_ttft_ms	steady_qk_ms	steady_routing_ms	steady_selected_ttft_ms	avg_full_ttft_ms
EOF

for spec in "${CASE_SPECS[@]}"; do
    IFS=':' read -r case_name gate_mode seg_size keep_ratio min_keep max_keep <<< "$spec"
    append_summary "$case_name" "$gate_mode" "$seg_size" "$keep_ratio" "$min_keep" "$max_keep"
done

{
    echo "============================================================"
    echo " QMSum Coarse Segment Gate Budget Sweep Summary"
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

exit "$OVERALL_STATUS"
