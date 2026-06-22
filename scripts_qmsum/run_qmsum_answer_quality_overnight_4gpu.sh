#!/bin/bash
# ============================================================================
# Overnight QMSum answer-quality sweep on up to four GPUs.
#
# Goal:
#   Run a broader but still controlled sweep after the answer-leak cleanup:
#   1. fixed vs dynamic candidate pool
#   2. answer length budget
#   3. evidence order / context format
#   4. prompt strictness
#   5. higher selected-context recall budget
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   START_DOC=0 END_DOC=30 MAX_QUERIES=5 GPU_LIST="0 1 2 3" \
#   bash scripts_qmsum/run_qmsum_answer_quality_overnight_4gpu.sh
# ============================================================================

set -u

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [ "$SCRIPT_DIR" = "$SCRIPT_PATH" ]; then
    SCRIPT_DIR="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"

START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-30}
MAX_QUERIES=${MAX_QUERIES:-5}
GPU_LIST=${GPU_LIST:-"0 1 2 3"}

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}
NUM_NODES=${NUM_NODES:-4}
HIER_TOP_TOPICS=${HIER_TOP_TOPICS:-1}
MAX_TOKENS=${MAX_TOKENS:-0}
USE_PER_HEAD=${USE_PER_HEAD:-1}

BASE_ROUTE_TOP_K=${BASE_ROUTE_TOP_K:-12}
QK_SCORE_BATCH_SIZE=${QK_SCORE_BATCH_SIZE:-64}
CACHE_CANDIDATE_KEYS=${CACHE_CANDIDATE_KEYS:-1}

DYNAMIC_SUMMARY_TOP_K=${DYNAMIC_SUMMARY_TOP_K:-16}
DYNAMIC_DETAIL_TOP_K=${DYNAMIC_DETAIL_TOP_K:-12}
DYNAMIC_BALANCED_TOP_K=${DYNAMIC_BALANCED_TOP_K:-12}
DYNAMIC_CANDIDATE_POOL_BUDGET_MAP=${DYNAMIC_CANDIDATE_POOL_BUDGET_MAP:-summary:96,detail:48,balanced:56,default:56}
DYNAMIC_CANDIDATE_POOL_MIN_KEEP=${DYNAMIC_CANDIDATE_POOL_MIN_KEEP:-24}

WIDE_DYNAMIC_SUMMARY_TOP_K=${WIDE_DYNAMIC_SUMMARY_TOP_K:-20}
WIDE_DYNAMIC_DETAIL_TOP_K=${WIDE_DYNAMIC_DETAIL_TOP_K:-16}
WIDE_DYNAMIC_BALANCED_TOP_K=${WIDE_DYNAMIC_BALANCED_TOP_K:-16}
WIDE_DYNAMIC_CANDIDATE_POOL_BUDGET_MAP=${WIDE_DYNAMIC_CANDIDATE_POOL_BUDGET_MAP:-summary:128,detail:64,balanced:72,default:72}

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

ANSWER_EVIDENCE_MAX_ENTRIES=${ANSWER_EVIDENCE_MAX_ENTRIES:-80}
ANSWER_EVIDENCE_MAX_CHARS=${ANSWER_EVIDENCE_MAX_CHARS:-600}

FETCH_BANDWIDTH_GBPS=${FETCH_BANDWIDTH_GBPS:-25.0}
PER_NODE_RTT_MS=${PER_NODE_RTT_MS:-1.0}
PER_SEGMENT_OVERHEAD_MS=${PER_SEGMENT_OVERHEAD_MS:-0.15}
DECODE_STARTUP_MS=${DECODE_STARTUP_MS:-15.0}
QUERY_TOKENIZER_WARMUP=${QUERY_TOKENIZER_WARMUP:-1}
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}

LOG_ROOT=${LOG_ROOT:-logs/qmsum_answer_quality_overnight_${START_DOC}_${END_DOC}_q${MAX_QUERIES}}
SUMMARY_TSV="$LOG_ROOT/summary.tsv"
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "GPU_LIST is empty. Example: GPU_LIST=\"0 1 2 3\""
    exit 1
fi

CASE_NAMES=(
    fixed_grounded_96
    dyn_grounded_96
    dyn_grounded_64
    dyn_grounded_48
    dyn_timeorder_64
    dyn_chunkturns_64
    dyn_strict_64
    dyn_top16_grounded_64
)

case_config() {
    local case_name="$1"

    CASE_DYNAMIC_POOL=1
    CASE_ROUTE_TOP_K="$BASE_ROUTE_TOP_K"
    CASE_DYNAMIC_SUMMARY_TOP_K="$DYNAMIC_SUMMARY_TOP_K"
    CASE_DYNAMIC_DETAIL_TOP_K="$DYNAMIC_DETAIL_TOP_K"
    CASE_DYNAMIC_BALANCED_TOP_K="$DYNAMIC_BALANCED_TOP_K"
    CASE_DYNAMIC_POOL_MAP="$DYNAMIC_CANDIDATE_POOL_BUDGET_MAP"
    CASE_ANSWER_MAX_NEW_TOKENS=64
    CASE_CONTEXT_MODE=turns
    CASE_EVIDENCE_ORDER=qk_then_time
    CASE_PROMPT_STYLE=grounded
    CASE_EVIDENCE_MAX_ENTRIES="$ANSWER_EVIDENCE_MAX_ENTRIES"
    CASE_EVIDENCE_MAX_CHARS="$ANSWER_EVIDENCE_MAX_CHARS"

    case "$case_name" in
        fixed_grounded_96)
            CASE_DYNAMIC_POOL=0
            CASE_ANSWER_MAX_NEW_TOKENS=96
            ;;
        dyn_grounded_96)
            CASE_ANSWER_MAX_NEW_TOKENS=96
            ;;
        dyn_grounded_64)
            CASE_ANSWER_MAX_NEW_TOKENS=64
            ;;
        dyn_grounded_48)
            CASE_ANSWER_MAX_NEW_TOKENS=48
            ;;
        dyn_timeorder_64)
            CASE_EVIDENCE_ORDER=time
            ;;
        dyn_chunkturns_64)
            CASE_CONTEXT_MODE=chunk_turns
            ;;
        dyn_strict_64)
            CASE_PROMPT_STYLE=strict
            ;;
        dyn_top16_grounded_64)
            CASE_ROUTE_TOP_K=16
            CASE_DYNAMIC_SUMMARY_TOP_K="$WIDE_DYNAMIC_SUMMARY_TOP_K"
            CASE_DYNAMIC_DETAIL_TOP_K="$WIDE_DYNAMIC_DETAIL_TOP_K"
            CASE_DYNAMIC_BALANCED_TOP_K="$WIDE_DYNAMIC_BALANCED_TOP_K"
            CASE_DYNAMIC_POOL_MAP="$WIDE_DYNAMIC_CANDIDATE_POOL_BUDGET_MAP"
            ;;
        *)
            echo "Unknown case: $case_name"
            return 1
            ;;
    esac
}

case_tag() {
    local case_name="$1"
    case_config "$case_name" || return 1
    echo "overnight_${case_name}_s${ROUTE_COARSE_SEGMENT_SIZE}_r${ROUTE_COARSE_SEGMENT_KEEP_RATIO}_m${ROUTE_COARSE_SEGMENT_MIN_KEEP}_k${CASE_ROUTE_TOP_K}_batch${QK_SCORE_BATCH_SIZE}_q${MAX_QUERIES}"
}

run_case() {
    local gpu_id="$1"
    local case_name="$2"
    case_config "$case_name" || return 1

    local tag
    tag="$(case_tag "$case_name")"
    local log_dir="$LOG_ROOT/$case_name"

    echo "[$(date '+%F %T')] START case=$case_name gpu=$gpu_id tag=$tag"

    MODEL_ID="$MODEL_ID" \
    DATA_PATH="$DATA_PATH" \
    NGPUS="$NGPUS" \
    MEM="$MEM" \
    GPU_ID="$gpu_id" \
    START_DOC="$START_DOC" \
    END_DOC="$END_DOC" \
    MAX_QUERIES="$MAX_QUERIES" \
    NUM_NODES="$NUM_NODES" \
    HIER_TOP_TOPICS="$HIER_TOP_TOPICS" \
    MAX_TOKENS="$MAX_TOKENS" \
    USE_PER_HEAD="$USE_PER_HEAD" \
    ROUTE_TOP_K="$CASE_ROUTE_TOP_K" \
    QK_SCORE_BATCH_SIZE="$QK_SCORE_BATCH_SIZE" \
    CACHE_CANDIDATE_KEYS="$CACHE_CANDIDATE_KEYS" \
    DYNAMIC_ROUTE_BUDGET=1 \
    DYNAMIC_SUMMARY_TOP_K="$CASE_DYNAMIC_SUMMARY_TOP_K" \
    DYNAMIC_DETAIL_TOP_K="$CASE_DYNAMIC_DETAIL_TOP_K" \
    DYNAMIC_BALANCED_TOP_K="$CASE_DYNAMIC_BALANCED_TOP_K" \
    DYNAMIC_CANDIDATE_POOL_BUDGET="$CASE_DYNAMIC_POOL" \
    DYNAMIC_CANDIDATE_POOL_BUDGET_MAP="$CASE_DYNAMIC_POOL_MAP" \
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
    ANSWER_EVIDENCE_ORDER="$CASE_EVIDENCE_ORDER" \
    SELECTED_ANSWER_CONTEXT_MODE="$CASE_CONTEXT_MODE" \
    ANSWER_PROMPT_STYLE="$CASE_PROMPT_STYLE" \
    ANSWER_MAX_NEW_TOKENS="$CASE_ANSWER_MAX_NEW_TOKENS" \
    ANSWER_EVIDENCE_MAX_ENTRIES="$CASE_EVIDENCE_MAX_ENTRIES" \
    ANSWER_EVIDENCE_MAX_CHARS="$CASE_EVIDENCE_MAX_CHARS" \
    FETCH_BANDWIDTH_GBPS="$FETCH_BANDWIDTH_GBPS" \
    PER_NODE_RTT_MS="$PER_NODE_RTT_MS" \
    PER_SEGMENT_OVERHEAD_MS="$PER_SEGMENT_OVERHEAD_MS" \
    DECODE_STARTUP_MS="$DECODE_STARTUP_MS" \
    QUERY_TOKENIZER_WARMUP="$QUERY_TOKENIZER_WARMUP" \
    CASE_SUMMARY_TAG="$tag" \
    LOG_DIR="$log_dir" \
    RESUME_IF_LOG_OK="$RESUME_IF_LOG_OK" \
    bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh"

    local status=$?
    echo "[$(date '+%F %T')] END case=$case_name gpu=$gpu_id status=$status"
    return "$status"
}

append_summary() {
    local case_name="$1"
    local status="$2"
    case_config "$case_name" || return 1

    local tag
    tag="$(case_tag "$case_name")"
    local case_tsv="outputs/qmsum_case_summary_N${NUM_NODES}_${START_DOC}_${END_DOC}_${tag}.tsv"
    local log_dir="$LOG_ROOT/$case_name"

    if [ "$status" -ne 0 ] || [ ! -f "$case_tsv" ]; then
        {
            printf "%s\tRUN_FAIL\t0" "$case_name"
            for _field in \
                avg_full_f1 avg_selected_f1 avg_oracle_f1 \
                avg_selected_minus_full avg_selected_minus_oracle \
                selected_ge_full_rate selected_ge_oracle_rate \
                full_bad_output_rate selected_bad_output_rate oracle_bad_output_rate \
                avg_turn_recall avg_turn_precision avg_turn_f1 \
                avg_qk_candidates avg_prefilter_prune avg_dynamic_pool_prune avg_gate_prune \
                avg_qk_ms avg_qk_total_ms avg_routing_overhead_ms avg_selected_ttft_ms \
                ctx_token_saving_pct avg_selected_kv_mib avg_selected_fetch_ms
            do
                printf "\tNA"
            done
            printf "\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$CASE_ROUTE_TOP_K" \
                "$CASE_DYNAMIC_POOL" \
                "$CASE_DYNAMIC_POOL_MAP" \
                "$CASE_ANSWER_MAX_NEW_TOKENS" \
                "$CASE_CONTEXT_MODE" \
                "$CASE_EVIDENCE_ORDER" \
                "$CASE_PROMPT_STYLE" \
                "$CASE_DYNAMIC_DETAIL_TOP_K" \
                "$case_tsv" \
                "$log_dir" \
                "$tag"
        } >> "$SUMMARY_TSV"
        return
    fi

    CASE_NAME="$case_name" \
    CASE_TSV="$case_tsv" \
    CASE_ROUTE_TOP_K="$CASE_ROUTE_TOP_K" \
    CASE_DYNAMIC_POOL="$CASE_DYNAMIC_POOL" \
    CASE_DYNAMIC_POOL_MAP="$CASE_DYNAMIC_POOL_MAP" \
    CASE_ANSWER_MAX_NEW_TOKENS="$CASE_ANSWER_MAX_NEW_TOKENS" \
    CASE_CONTEXT_MODE="$CASE_CONTEXT_MODE" \
    CASE_EVIDENCE_ORDER="$CASE_EVIDENCE_ORDER" \
    CASE_PROMPT_STYLE="$CASE_PROMPT_STYLE" \
    CASE_DYNAMIC_DETAIL_TOP_K="$CASE_DYNAMIC_DETAIL_TOP_K" \
    LOG_DIR_CASE="$log_dir" \
    CASE_TAG="$tag" \
    python - <<'PY' >> "$SUMMARY_TSV"
import csv
import os

case_name = os.environ["CASE_NAME"]
case_tsv = os.environ["CASE_TSV"]

with open(case_tsv, encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

def as_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default

def avg(field):
    vals = [as_float(row.get(field)) for row in rows if row.get(field, "") != ""]
    return sum(vals) / len(vals) if vals else 0.0

def rate(field):
    if not rows:
        return 0.0
    return sum(1 for row in rows if str(row.get(field, "")).strip() == "1") / len(rows)

def compare_rate(left, right):
    vals = [
        as_float(row.get(left)) >= as_float(row.get(right))
        for row in rows
        if row.get(left, "") != "" and row.get(right, "") != ""
    ]
    return sum(vals) / len(vals) if vals else 0.0

values = [
    case_name,
    "OK",
    str(len(rows)),
    f"{avg('full_answer_f1'):.4f}",
    f"{avg('selected_answer_f1'):.4f}",
    f"{avg('oracle_answer_f1'):.4f}",
    f"{avg('answer_f1_delta'):.4f}",
    f"{avg('selected_answer_f1_delta_vs_oracle'):.4f}",
    f"{compare_rate('selected_answer_f1', 'full_answer_f1'):.4f}",
    f"{compare_rate('selected_answer_f1', 'oracle_answer_f1'):.4f}",
    f"{rate('full_bad_output'):.4f}",
    f"{rate('selected_bad_output'):.4f}",
    f"{rate('oracle_bad_output'):.4f}",
    f"{avg('selected_turn_recall'):.4f}",
    f"{avg('selected_turn_precision'):.4f}",
    f"{avg('selected_turn_f1'):.4f}",
    f"{avg('num_candidates_scored_qk'):.1f}",
    f"{avg('candidate_prefilter_prune_ratio'):.4f}",
    f"{avg('dynamic_candidate_pool_prune_ratio'):.4f}",
    f"{avg('coarse_segment_gate_prune_ratio'):.4f}",
    f"{avg('qk_scoring_ms'):.2f}",
    f"{avg('qk_total_stage_ms'):.2f}",
    f"{avg('routing_overhead_ms'):.2f}",
    f"{avg('selected_ttft_ms'):.2f}",
    f"{avg('ctx_token_saving_pct'):.1f}",
    f"{avg('selected_kv_mib'):.2f}",
    f"{avg('selected_fetch_latency_ms'):.2f}",
    os.environ["CASE_ROUTE_TOP_K"],
    os.environ["CASE_DYNAMIC_POOL"],
    os.environ["CASE_DYNAMIC_POOL_MAP"],
    os.environ["CASE_ANSWER_MAX_NEW_TOKENS"],
    os.environ["CASE_CONTEXT_MODE"],
    os.environ["CASE_EVIDENCE_ORDER"],
    os.environ["CASE_PROMPT_STYLE"],
    os.environ["CASE_DYNAMIC_DETAIL_TOP_K"],
    case_tsv,
    os.environ["LOG_DIR_CASE"],
    os.environ["CASE_TAG"],
]
print("\t".join(values))
PY
}

write_summary_header() {
    cat > "$SUMMARY_TSV" <<'EOF'
case_name	status	n	avg_full_f1	avg_selected_f1	avg_oracle_f1	avg_selected_minus_full	avg_selected_minus_oracle	selected_ge_full_rate	selected_ge_oracle_rate	full_bad_output_rate	selected_bad_output_rate	oracle_bad_output_rate	avg_turn_recall	avg_turn_precision	avg_turn_f1	avg_qk_candidates	avg_prefilter_prune	avg_dynamic_pool_prune	avg_gate_prune	avg_qk_ms	avg_qk_total_ms	avg_routing_overhead_ms	avg_selected_ttft_ms	ctx_token_saving_pct	avg_selected_kv_mib	avg_selected_fetch_ms	route_top_k	dynamic_pool	dynamic_pool_map	answer_max_new_tokens	context_mode	evidence_order	prompt_style	dynamic_detail_top_k	case_tsv	log_dir	case_tag
EOF
}

write_summary_view() {
    {
        echo "============================================================"
        echo " QMSum Answer Quality Overnight Summary"
        echo "============================================================"
        echo "docs=$START_DOC:$END_DOC  max_queries=$MAX_QUERIES  gpu_list=$GPU_LIST"
        echo "summary_tsv=$SUMMARY_TSV"
        echo ""
        if command -v column >/dev/null 2>&1; then
            column -t -s $'\t' "$SUMMARY_TSV"
        else
            cat "$SUMMARY_TSV"
        fi
        echo ""
        echo "How to read the main columns:"
        echo "  avg_selected_f1 / avg_full_f1: selective answer quality vs full-context baseline"
        echo "  avg_oracle_f1: answer generator with gold relevant turns; upper-bound diagnostic"
        echo "  selected_bad_output_rate: prompt leak/repetition/web-like bad output detector"
        echo "  avg_qk_candidates and avg_qk_total_ms: exact-QK workload and measured routing time"
        echo "  avg_selected_ttft_ms: routing + estimated selected KV fetch + decode startup"
        echo "  ctx_token_saving_pct: answer prompt context-token reduction vs full prompt"
        echo ""
        echo "Logs live under: $LOG_ROOT"
    } | tee "$SUMMARY_TXT"
}

echo "============================================================"
echo " QMSum Answer Quality Overnight 4-GPU Sweep"
echo " docs=$START_DOC:$END_DOC"
echo " MAX_QUERIES=$MAX_QUERIES"
echo " GPU_LIST=$GPU_LIST"
echo " LOG_ROOT=$LOG_ROOT"
echo " base dynamic pool map=$DYNAMIC_CANDIDATE_POOL_BUDGET_MAP"
echo " wide dynamic pool map=$WIDE_DYNAMIC_CANDIDATE_POOL_BUDGET_MAP"
echo " cases=${CASE_NAMES[*]}"
echo "============================================================"

write_summary_header

overall_status=0
case_count=${#CASE_NAMES[@]}
gpu_count=${#GPUS[@]}

start=0
while [ "$start" -lt "$case_count" ]; do
    pids=()
    names=()
    echo ""
    echo "---- Launching wave starting at case index $start ----"

    slot=0
    while [ "$slot" -lt "$gpu_count" ] && [ $((start + slot)) -lt "$case_count" ]; do
        case_name="${CASE_NAMES[$((start + slot))]}"
        gpu_id="${GPUS[$slot]}"
        run_case "$gpu_id" "$case_name" &
        pids+=("$!")
        names+=("$case_name")
        slot=$((slot + 1))
    done

    idx=0
    while [ "$idx" -lt "${#pids[@]}" ]; do
        pid="${pids[$idx]}"
        case_name="${names[$idx]}"
        if wait "$pid"; then
            status=0
        else
            status=$?
            overall_status=1
        fi
        append_summary "$case_name" "$status"
        idx=$((idx + 1))
    done

    write_summary_view
    start=$((start + gpu_count))
done

write_summary_view

if [ "$overall_status" -ne 0 ]; then
    echo "At least one case failed. Check $SUMMARY_TSV and logs under $LOG_ROOT."
    exit 1
fi

echo "All overnight cases finished successfully."
