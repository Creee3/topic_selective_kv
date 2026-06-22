#!/bin/bash
# ============================================================================
# Compare topic embedding with and without topic label signal.
#
# Purpose:
#   Verify whether the topic title helps the top-level embedding router.
#
# Cases:
#   1. label_weight = 0.0
#   2. label_weight = 0.35 (current default)
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash run_qmsum_topic_label_ablation.sh
#
# Optional overrides:
#   START_DOC=0 END_DOC=5 MAX_QUERIES=2 \
#   TOPIC_EMBEDDING_TURN_SCORE_MODE=topk_mean TOPIC_EMBEDDING_TOPK=3 \
#   bash run_qmsum_topic_label_ablation.sh
#
# Outputs:
#   logs/qmsum_topic_label_ablation/*.log
#   logs/qmsum_topic_label_ablation/summary.tsv
#   logs/qmsum_topic_label_ablation/summary.txt
# ============================================================================

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}
START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-5}
MAX_QUERIES=${MAX_QUERIES:-2}
HIER_TOP_TOPICS=${HIER_TOP_TOPICS:-1}
HIER_TOPIC_SCORE_MODE=${HIER_TOPIC_SCORE_MODE:-sum}
TOPIC_EMBEDDING_TURN_SCORE_MODE=${TOPIC_EMBEDDING_TURN_SCORE_MODE:-topk_mean}
TOPIC_EMBEDDING_TOPK=${TOPIC_EMBEDDING_TOPK:-3}
ROUTE_TOP_K=${ROUTE_TOP_K:-16}
USE_PER_HEAD=${USE_PER_HEAD:-1}
LOG_DIR=${LOG_DIR:-logs/qmsum_topic_label_ablation}
SUMMARY_TSV="$LOG_DIR/summary.tsv"
SUMMARY_TXT="$LOG_DIR/summary.txt"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " QMSum Topic Label Ablation"
echo " model=$MODEL_ID"
echo " data=$DATA_PATH"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries_per_doc=$MAX_QUERIES"
echo " hier_top_topics=$HIER_TOP_TOPICS"
echo " hier_topic_score_mode=$HIER_TOPIC_SCORE_MODE"
echo " topic_embedding_turn_score_mode=$TOPIC_EMBEDDING_TURN_SCORE_MODE"
echo " topic_embedding_topk=$TOPIC_EMBEDDING_TOPK"
echo " route_chunk_size=128"
echo " route_top_k=$ROUTE_TOP_K"
echo " route_per_head=$USE_PER_HEAD"
echo " logs=$LOG_DIR"
echo "============================================================"

cat > "$SUMMARY_TSV" <<EOF
case	status	topic_label_weight	top1_pct	top2_pct	selected_topic_hit_pct	selected_turn_hit_pct	avg_turn_recall_pct	avg_turn_precision_pct	avg_turn_f1_pct	avg_transfer_segments	avg_coalescing_gain_pct
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

append_summary() {
    local case_name="$1"
    local topic_label_weight="$2"
    local status="$3"
    local log_file="$4"

    if [ "$status" != "OK" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "$status" "$topic_label_weight" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    if ! grep -Fq "QMSum routing summary" "$log_file"; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "PARSE_FAIL" "$topic_label_weight" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    local strategy_line selected_topic_line selected_turn_line
    local recall_line precision_line f1_line transfer_segments_line coalescing_line
    local top1_pct top2_pct selected_topic_pct selected_turn_pct
    local recall_pct precision_pct f1_pct transfer_segments coalescing_pct

    strategy_line=$(extract_strategy_line "  Relevant-span topic hit rate \\(loose\\):" "embedding top-1:" "$log_file")
    selected_topic_line=$(extract_line "selected-topic hit (embedding):" "$log_file")
    selected_turn_line=$(extract_line "selected-turn hit:" "$log_file")
    recall_line=$(extract_line "avg turn recall:" "$log_file")
    precision_line=$(extract_line "avg turn precision:" "$log_file")
    f1_line=$(extract_line "avg turn F1:" "$log_file")
    transfer_segments_line=$(extract_line "avg transfer segments:" "$log_file")
    coalescing_line=$(extract_line "avg coalescing gain:" "$log_file")

    if [ -z "$strategy_line" ] || [ -z "$selected_topic_line" ] || [ -z "$selected_turn_line" ] || \
       [ -z "$recall_line" ] || [ -z "$precision_line" ] || [ -z "$f1_line" ] || \
       [ -z "$transfer_segments_line" ] || [ -z "$coalescing_line" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "PARSE_FAIL" "$topic_label_weight" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    top1_pct=$(echo "$strategy_line" | sed -E 's/.*\(([0-9.]+)%\), top-2: .*/\1/')
    top2_pct=$(echo "$strategy_line" | sed -E 's/.*top-2: [0-9]+\/[0-9]+ \(([0-9.]+)%\).*/\1/')
    selected_topic_pct=$(echo "$selected_topic_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    selected_turn_pct=$(echo "$selected_turn_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    recall_pct=$(echo "$recall_line" | awk '{print $4}' | tr -d '%')
    precision_pct=$(echo "$precision_line" | awk '{print $4}' | tr -d '%')
    f1_pct=$(echo "$f1_line" | awk '{print $4}' | tr -d '%')
    transfer_segments=$(echo "$transfer_segments_line" | awk '{print $4}')
    coalescing_pct=$(echo "$coalescing_line" | awk '{print $4}' | tr -d '%')

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$case_name" "OK" "$topic_label_weight" \
        "$top1_pct" "$top2_pct" "$selected_topic_pct" "$selected_turn_pct" \
        "$recall_pct" "$precision_pct" "$f1_pct" "$transfer_segments" "$coalescing_pct" >> "$SUMMARY_TSV"
}

run_case() {
    local case_name="$1"
    local topic_label_weight="$2"
    local log_file="$LOG_DIR/${case_name}.log"
    local extra_args=()

    echo ""
    echo "============================================================"
    echo " Running case: $case_name"
    echo " topic_label_weight: $topic_label_weight"
    echo " Log file: $log_file"
    echo "============================================================"

    if [ "$USE_PER_HEAD" -eq 1 ]; then
        extra_args+=(--route_per_head)
    fi

    python qmsum_sim.py \
        --data_path "$DATA_PATH" \
        --model_id "$MODEL_ID" \
        --num_gpus "$NGPUS" \
        --max_gpu_memory "$MEM" \
        --start_doc "$START_DOC" \
        --end_doc "$END_DOC" \
        --max_queries_per_doc "$MAX_QUERIES" \
        --baselines \
        --routing_granularity hierarchical \
        --hier_top_topics "$HIER_TOP_TOPICS" \
        --hier_top_strategy embedding \
        --hier_topic_score_mode "$HIER_TOPIC_SCORE_MODE" \
        --topic_embedding_turn_score_mode "$TOPIC_EMBEDDING_TURN_SCORE_MODE" \
        --topic_embedding_topk "$TOPIC_EMBEDDING_TOPK" \
        --topic_label_weight "$topic_label_weight" \
        --route_chunk_size 128 \
        --route_top_k "$ROUTE_TOP_K" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    local pipe_status=${PIPESTATUS[0]}

    if [ "$pipe_status" -ne 0 ]; then
        echo " Case failed with exit code: $pipe_status" | tee -a "$log_file"
        append_summary "$case_name" "$topic_label_weight" "RUN_FAIL" "$log_file"
        return
    fi

    append_summary "$case_name" "$topic_label_weight" "OK" "$log_file"
}

run_case "label_0.0" "0.0"
run_case "label_0.35" "0.35"

{
    echo "============================================================"
    echo " QMSum Topic Label Ablation Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Ablation complete"
echo " Check logs under: $LOG_DIR"
echo " Key summary: $SUMMARY_TXT"
echo "============================================================"
