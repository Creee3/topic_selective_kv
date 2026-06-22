#!/bin/bash
# ============================================================================
# Compare fine-stage neighbor-chunk expansion on the current mainline.
#
# Fixed mainline:
#   lexical coarse routing
#   -> top1 topic
#   -> Q-K chunk routing
#   -> route_top_k=12
#
# Sweep:
#   route_neighbor_expand = 0 / 1 / 2
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_neighbor_expand_compare.sh
#
# Optional overrides:
#   START_DOC=5 END_DOC=10 MAX_QUERIES=1 GPU_ID=2 \
#   bash scripts_qmsum/run_qmsum_neighbor_expand_compare.sh
# ============================================================================

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}
GPU_ID=${GPU_ID:-0}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-10}
MAX_QUERIES=${MAX_QUERIES:-1}
CHUNK_SIZE=${CHUNK_SIZE:-128}
ROUTE_TOP_K=${ROUTE_TOP_K:-12}
USE_PER_HEAD=${USE_PER_HEAD:-1}
ANSWER_MAX_NEW_TOKENS=${ANSWER_MAX_NEW_TOKENS:-96}
NEIGHBOR_LIST_STR=${NEIGHBOR_LIST:-"0 1 2"}
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}
LOG_DIR=${LOG_DIR:-logs/qmsum_neighbor_expand_compare}
SUMMARY_TSV="$LOG_DIR/summary.tsv"
SUMMARY_TXT="$LOG_DIR/summary.txt"

read -r -a NEIGHBOR_LIST <<< "$NEIGHBOR_LIST_STR"

mkdir -p "$LOG_DIR"

cat > "$SUMMARY_TSV" <<EOF
neighbor_expand	status	selected_topic_hit_pct	selected_turn_hit_pct	avg_turn_recall_pct	avg_turn_precision_pct	avg_turn_f1_pct	full_answer_f1_pct	selected_answer_f1_pct	answer_f1_delta_pct	ctx_token_saving_pct
EOF

extract_line() {
    local pattern="$1"
    local file="$2"
    grep -F "$pattern" "$file" | tail -n 1
}

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

append_summary() {
    local neighbor_expand="$1"
    local status="$2"
    local log_file="$3"

    if [ "$status" != "OK" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$neighbor_expand" "$status" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    local selected_topic_line selected_turn_line recall_line precision_line f1_line
    local full_answer_line selected_answer_line answer_delta_line ctx_saving_line

    selected_topic_line=$(extract_line "selected-topic hit (lexical):" "$log_file")
    selected_turn_line=$(extract_line "selected-turn hit:" "$log_file")
    recall_line=$(extract_line "avg turn recall:" "$log_file")
    precision_line=$(extract_line "avg turn precision:" "$log_file")
    f1_line=$(extract_line "avg turn F1:" "$log_file")
    full_answer_line=$(extract_line "avg full-answer F1:" "$log_file")
    selected_answer_line=$(extract_line "avg selective-answer F1:" "$log_file")
    answer_delta_line=$(extract_line "avg F1 delta:" "$log_file")
    ctx_saving_line=$(extract_line "avg ctx token saving:" "$log_file")

    if [ -z "$selected_topic_line" ] || [ -z "$selected_turn_line" ] || \
       [ -z "$recall_line" ] || [ -z "$precision_line" ] || [ -z "$f1_line" ] || \
       [ -z "$full_answer_line" ] || [ -z "$selected_answer_line" ] || \
       [ -z "$answer_delta_line" ] || [ -z "$ctx_saving_line" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$neighbor_expand" "PARSE_FAIL" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    local selected_topic_pct selected_turn_pct recall_pct precision_pct f1_pct
    local full_answer_pct selected_answer_pct answer_delta_pct ctx_saving_pct

    selected_topic_pct=$(echo "$selected_topic_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    selected_turn_pct=$(echo "$selected_turn_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    recall_pct=$(echo "$recall_line" | awk '{print $4}' | tr -d '%')
    precision_pct=$(echo "$precision_line" | awk '{print $4}' | tr -d '%')
    f1_pct=$(echo "$f1_line" | awk '{print $4}' | tr -d '%')
    full_answer_pct=$(echo "$full_answer_line" | awk '{print $4}' | tr -d '%')
    selected_answer_pct=$(echo "$selected_answer_line" | awk '{print $4}' | tr -d '%')
    answer_delta_pct=$(echo "$answer_delta_line" | awk '{print $4}' | tr -d '%')
    ctx_saving_pct=$(echo "$ctx_saving_line" | awk '{print $5}' | tr -d '%')

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$neighbor_expand" "OK" \
        "$selected_topic_pct" "$selected_turn_pct" "$recall_pct" "$precision_pct" "$f1_pct" \
        "$full_answer_pct" "$selected_answer_pct" "$answer_delta_pct" "$ctx_saving_pct" >> "$SUMMARY_TSV"
}

run_case() {
    local neighbor_expand="$1"
    local case_name="neighbor_${neighbor_expand}"
    local log_file="$LOG_DIR/${case_name}.log"
    local extra_args=()

    if [ "$RESUME_IF_LOG_OK" -eq 1 ] && has_complete_log "$log_file"; then
        append_summary "$neighbor_expand" "OK" "$log_file"
        return
    fi

    if [ "$USE_PER_HEAD" -eq 1 ]; then
        extra_args+=(--route_per_head)
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" python qmsum_mainline.py \
        --data_path "$DATA_PATH" \
        --model_id "$MODEL_ID" \
        --num_gpus "$NGPUS" \
        --max_gpu_memory "$MEM" \
        --start_doc "$START_DOC" \
        --end_doc "$END_DOC" \
        --max_queries_per_doc "$MAX_QUERIES" \
        --hier_top_topics 1 \
        --hier_topic_score_mode sum \
        --route_chunk_size "$CHUNK_SIZE" \
        --route_top_k "$ROUTE_TOP_K" \
        --route_neighbor_expand "$neighbor_expand" \
        --eval_answers \
        --answer_max_new_tokens "$ANSWER_MAX_NEW_TOKENS" \
        --case_summary_tag "neighbor_expand_${neighbor_expand}" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    local pipe_status=${PIPESTATUS[0]}

    if [ "$pipe_status" -ne 0 ]; then
        append_summary "$neighbor_expand" "RUN_FAIL" "$log_file"
        return
    fi

    append_summary "$neighbor_expand" "OK" "$log_file"
}

for neighbor_expand in "${NEIGHBOR_LIST[@]}"; do
    run_case "$neighbor_expand"
done

{
    echo "============================================================"
    echo " QMSum Neighbor-Expand Compare Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Compare complete"
echo " Check logs under: $LOG_DIR"
echo " Key summary: $SUMMARY_TXT"
echo "============================================================"
