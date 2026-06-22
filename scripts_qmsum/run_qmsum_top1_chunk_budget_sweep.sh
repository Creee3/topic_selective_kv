#!/bin/bash
# ============================================================================
# Focused sweep for the current mainline hierarchical QMSum setting.
#
# Purpose:
#   Keep top-level routing fixed at 1 topic and only sweep chunk budget.
#   This reflects the current best direction:
#     lexical topic routing
#     -> 1 topic
#     -> Q-K chunk routing with different route_top_k budgets
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash run_qmsum_top1_chunk_budget_sweep.sh
#
# Optional overrides:
#   CHUNK_TOPK_LIST="4 8 12 16" START_DOC=0 END_DOC=5 MAX_QUERIES=2 \
#     bash run_qmsum_top1_chunk_budget_sweep.sh
#
# Outputs:
#   logs/qmsum_top1_chunk_budget_sweep/*.log
#   logs/qmsum_top1_chunk_budget_sweep/summary.tsv
#   logs/qmsum_top1_chunk_budget_sweep/summary.txt
# ============================================================================

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}
START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-5}
MAX_QUERIES=${MAX_QUERIES:-2}
CHUNK_TOPK_LIST_STR=${CHUNK_TOPK_LIST:-"4 8 12 16"}
HIER_TOP_TOPICS=${HIER_TOP_TOPICS:-1}
HIER_TOP_STRATEGY=${HIER_TOP_STRATEGY:-lexical}
HIER_TOPIC_SCORE_MODE=${HIER_TOPIC_SCORE_MODE:-sum}
CHUNK_SIZE=${CHUNK_SIZE:-128}
USE_PER_HEAD=${USE_PER_HEAD:-1}
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}
LOG_DIR=${LOG_DIR:-logs/qmsum_top1_chunk_budget_sweep}
SUMMARY_TSV="$LOG_DIR/summary.tsv"
SUMMARY_TXT="$LOG_DIR/summary.txt"

read -r -a CHUNK_TOPK_LIST <<< "$CHUNK_TOPK_LIST_STR"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " QMSum Top1-Topic Chunk-Budget Sweep"
echo " model=$MODEL_ID"
echo " data=$DATA_PATH"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries_per_doc=$MAX_QUERIES"
echo " hier_top_topics=$HIER_TOP_TOPICS"
echo " hier_top_strategy=$HIER_TOP_STRATEGY"
echo " hier_topic_score_mode=$HIER_TOPIC_SCORE_MODE"
echo " route_top_k=${CHUNK_TOPK_LIST[*]}"
echo " chunk_size=$CHUNK_SIZE"
echo " route_per_head=$USE_PER_HEAD"
echo " resume_if_log_ok=$RESUME_IF_LOG_OK"
echo " logs=$LOG_DIR"
echo "============================================================"

cat > "$SUMMARY_TSV" <<EOF
case	status	route_top_k	coarse_strategy	top1_pct	top2_pct	selected_topic_hit_pct	selected_turn_hit_pct	avg_turn_recall_pct	avg_turn_precision_pct	avg_turn_f1_pct
EOF

extract_line() {
    local pattern="$1"
    local file="$2"
    grep -F "$pattern" "$file" | tail -n 1
}

extract_strategy_line() {
    local section_header="$1"
    local strategy_name="$2"
    local file="$3"
    awk -v header="$section_header" -v strategy="$strategy_name" '
        $0 ~ header {flag=1; next}
        flag && /^  [A-Z]/ {flag=0}
        flag && $0 ~ strategy {print; exit}
    ' "$file"
}

has_complete_log() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 1
    fi
    grep -Fq "QMSum routing summary" "$file" || return 1
    grep -Fq "Saved to outputs/" "$file" || return 1
    return 0
}

append_summary() {
    local status="$1"
    local case_name="$2"
    local route_top_k="$3"
    local log_file="$4"

    if [ "$status" != "OK" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "$status" "$route_top_k" "$HIER_TOP_STRATEGY" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    if ! grep -Fq "QMSum routing summary" "$log_file"; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "PARSE_FAIL" "$route_top_k" "$HIER_TOP_STRATEGY" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    local strategy_line selected_topic_line selected_turn_line
    local recall_line precision_line f1_line
    local top1_pct top2_pct selected_topic_pct selected_turn_pct
    local recall_pct precision_pct f1_pct

    strategy_line=$(extract_strategy_line "  Relevant-span topic hit rate \\(loose\\):" "${HIER_TOP_STRATEGY} top-1:" "$log_file")
    selected_topic_line=$(extract_line "selected-topic hit (${HIER_TOP_STRATEGY}):" "$log_file")
    selected_turn_line=$(extract_line "selected-turn hit:" "$log_file")
    recall_line=$(extract_line "avg turn recall:" "$log_file")
    precision_line=$(extract_line "avg turn precision:" "$log_file")
    f1_line=$(extract_line "avg turn F1:" "$log_file")

    if [ -z "$strategy_line" ] || [ -z "$selected_topic_line" ] || [ -z "$selected_turn_line" ] || \
       [ -z "$recall_line" ] || [ -z "$precision_line" ] || [ -z "$f1_line" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "PARSE_FAIL" "$route_top_k" "$HIER_TOP_STRATEGY" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    top1_pct=$(echo "$strategy_line" | sed -E 's/.*\(([0-9.]+)%\), top-2: .*/\1/')
    top2_pct=$(echo "$strategy_line" | sed -E 's/.*top-2: [0-9]+\/[0-9]+ \(([0-9.]+)%\).*/\1/')
    selected_topic_pct=$(echo "$selected_topic_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    selected_turn_pct=$(echo "$selected_turn_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    recall_pct=$(echo "$recall_line" | awk '{print $4}' | tr -d '%')
    precision_pct=$(echo "$precision_line" | awk '{print $4}' | tr -d '%')
    f1_pct=$(echo "$f1_line" | awk '{print $4}' | tr -d '%')

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$case_name" "OK" "$route_top_k" "$HIER_TOP_STRATEGY" \
        "$top1_pct" "$top2_pct" "$selected_topic_pct" \
        "$selected_turn_pct" "$recall_pct" "$precision_pct" "$f1_pct" >> "$SUMMARY_TSV"
}

run_case() {
    local route_top_k="$1"
    local case_name="top1_chunks_${route_top_k}"
    local log_file="$LOG_DIR/${case_name}.log"
    local extra_args=()

    echo ""
    echo "============================================================"
    echo " Running case: $case_name"
    echo " hier_top_topics: $HIER_TOP_TOPICS"
    echo " route_top_k: $route_top_k"
    echo " Log file: $log_file"
    echo "============================================================"

    if [ "$RESUME_IF_LOG_OK" -eq 1 ] && has_complete_log "$log_file"; then
        echo " Existing complete log detected. Skip rerun and reuse: $log_file"
        append_summary "OK" "$case_name" "$route_top_k" "$log_file"
        return
    fi

    if [ "$USE_PER_HEAD" -eq 1 ]; then
        extra_args+=(--route_per_head)
    fi

    python qmsum_mainline.py \
        --data_path "$DATA_PATH" \
        --model_id "$MODEL_ID" \
        --num_gpus "$NGPUS" \
        --max_gpu_memory "$MEM" \
        --start_doc "$START_DOC" \
        --end_doc "$END_DOC" \
        --max_queries_per_doc "$MAX_QUERIES" \
        --hier_top_topics "$HIER_TOP_TOPICS" \
        --hier_top_strategy "$HIER_TOP_STRATEGY" \
        --hier_topic_score_mode "$HIER_TOPIC_SCORE_MODE" \
        --case_summary_tag "${HIER_TOP_STRATEGY}_${case_name}" \
        --route_chunk_size "$CHUNK_SIZE" \
        --route_top_k "$route_top_k" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    local pipe_status=${PIPESTATUS[0]}

    if [ "$pipe_status" -ne 0 ]; then
        echo " Case failed with exit code: $pipe_status" | tee -a "$log_file"
        append_summary "RUN_FAIL" "$case_name" "$route_top_k" "$log_file"
        return
    fi

    append_summary "OK" "$case_name" "$route_top_k" "$log_file"
}

for route_top_k in "${CHUNK_TOPK_LIST[@]}"; do
    run_case "$route_top_k"
done

{
    echo "============================================================"
    echo " QMSum Top1-Topic Chunk-Budget Sweep Summary"
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
echo " Check logs under: $LOG_DIR"
echo " Key summary: $SUMMARY_TXT"
echo "============================================================"
