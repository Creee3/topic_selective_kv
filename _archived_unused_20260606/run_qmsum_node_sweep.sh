#!/bin/bash
# ============================================================================
# One-click node-count sweep for QMSum routing experiments.
#
# Purpose:
#   Run QMSum routing with contiguous node assignment for several node counts,
#   then collect the stricter evidence-quality metrics into one summary.
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash run_qmsum_node_sweep.sh
#
# Optional overrides:
#   MODEL_ID=~/models/mistral-7b/ bash run_qmsum_node_sweep.sh
#   START_DOC=0 END_DOC=5 MAX_QUERIES=2 NODE_LIST="4 8 16" bash run_qmsum_node_sweep.sh
#
# Outputs:
#   logs/qmsum_node_sweep/*.log
#   logs/qmsum_node_sweep/summary.tsv
#   logs/qmsum_node_sweep/summary.txt
# ============================================================================

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}
START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-5}
MAX_QUERIES=${MAX_QUERIES:-2}
NODE_LIST_STR=${NODE_LIST:-"4 8 16"}
ROUTING_GRANULARITY=${ROUTING_GRANULARITY:-chunk}
NODE_ASSIGNMENT_MODE=${NODE_ASSIGNMENT_MODE:-contiguous}
CHUNK_SIZE=${CHUNK_SIZE:-128}
ROUTE_TOP_K=${ROUTE_TOP_K:-4}
USE_PER_HEAD=${USE_PER_HEAD:-1}
LOG_DIR=${LOG_DIR:-logs/qmsum_node_sweep}
SUMMARY_TSV="$LOG_DIR/summary.tsv"
SUMMARY_TXT="$LOG_DIR/summary.txt"

read -r -a NODE_LIST <<< "$NODE_LIST_STR"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " QMSum Node Sweep"
echo " model=$MODEL_ID"
echo " data=$DATA_PATH"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries_per_doc=$MAX_QUERIES"
echo " nodes=${NODE_LIST[*]}"
echo " routing_granularity=$ROUTING_GRANULARITY"
echo " node_assignment_mode=$NODE_ASSIGNMENT_MODE"
echo " chunk_size=$CHUNK_SIZE"
echo " route_top_k=$ROUTE_TOP_K"
echo " route_per_head=$USE_PER_HEAD"
echo " logs=$LOG_DIR"
echo "============================================================"

cat > "$SUMMARY_TSV" <<EOF
case	nodes	avg_relevant_nodes	all_node_span_pct	qk_loose_top1_pct	qk_loose_top2_pct	qk_dom_top1_pct	qk_dom_top2_pct	selected_turn_hit_pct	avg_turn_recall_pct	avg_turn_precision_pct	avg_turn_f1_pct
EOF

extract_line() {
    local pattern="$1"
    local file="$2"
    grep -F "$pattern" "$file" | tail -n 1
}

extract_qk_line_in_section() {
    local section_header="$1"
    local file="$2"
    awk -v header="$section_header" '
        $0 ~ header {flag=1; next}
        flag && /^  [A-Z]/ {flag=0}
        flag && /qk top-1:/ {print; exit}
    ' "$file"
}

append_summary() {
    local case_name="$1"
    local num_nodes="$2"
    local log_file="$3"

    local avg_nodes_line all_node_line
    local loose_qk_line dom_qk_line
    local selected_turn_line recall_line precision_line f1_line
    local avg_nodes all_node_pct
    local qk_loose_top1 qk_loose_top2 qk_dom_top1 qk_dom_top2
    local selected_turn_pct recall_pct precision_pct f1_pct

    avg_nodes_line=$(extract_line "avg relevant nodes:" "$log_file")
    all_node_line=$(extract_line "all-node relevant span:" "$log_file")
    loose_qk_line=$(extract_qk_line_in_section "  Relevant-span node hit rate \\(loose\\):" "$log_file")
    dom_qk_line=$(extract_qk_line_in_section "  Dominant-node hit rate \\(stricter\\):" "$log_file")
    selected_turn_line=$(extract_line "selected-turn hit:" "$log_file")
    recall_line=$(extract_line "avg turn recall:" "$log_file")
    precision_line=$(extract_line "avg turn precision:" "$log_file")
    f1_line=$(extract_line "avg turn F1:" "$log_file")

    avg_nodes=$(echo "$avg_nodes_line" | awk '{print $4}')
    all_node_pct=$(echo "$all_node_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')

    qk_loose_top1=$(echo "$loose_qk_line" | sed -E 's/.*\(([0-9.]+)%\), top-2: .*/\1/')
    qk_loose_top2=$(echo "$loose_qk_line" | sed -E 's/.*top-2: [0-9]+\/[0-9]+ \(([0-9.]+)%\).*/\1/')
    qk_dom_top1=$(echo "$dom_qk_line" | sed -E 's/.*\(([0-9.]+)%\), top-2: .*/\1/')
    qk_dom_top2=$(echo "$dom_qk_line" | sed -E 's/.*top-2: [0-9]+\/[0-9]+ \(([0-9.]+)%\).*/\1/')

    selected_turn_pct=$(echo "$selected_turn_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    recall_pct=$(echo "$recall_line" | awk '{print $4}' | tr -d '%')
    precision_pct=$(echo "$precision_line" | awk '{print $4}' | tr -d '%')
    f1_pct=$(echo "$f1_line" | awk '{print $4}' | tr -d '%')

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$case_name" "$num_nodes" "$avg_nodes" "$all_node_pct" \
        "$qk_loose_top1" "$qk_loose_top2" "$qk_dom_top1" "$qk_dom_top2" \
        "$selected_turn_pct" "$recall_pct" "$precision_pct" "$f1_pct" >> "$SUMMARY_TSV"
}

run_case() {
    local num_nodes="$1"
    local case_name="nodes_${num_nodes}"
    local log_file="$LOG_DIR/${case_name}.log"
    local extra_args=()

    echo ""
    echo "============================================================"
    echo " Running case: $case_name"
    echo " nodes: $num_nodes"
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
        --num_nodes "$num_nodes" \
        --start_doc "$START_DOC" \
        --end_doc "$END_DOC" \
        --max_queries_per_doc "$MAX_QUERIES" \
        --baselines \
        --routing_granularity "$ROUTING_GRANULARITY" \
        --route_chunk_size "$CHUNK_SIZE" \
        --route_top_k "$ROUTE_TOP_K" \
        --node_assignment_mode "$NODE_ASSIGNMENT_MODE" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"

    append_summary "$case_name" "$num_nodes" "$log_file"
}

for num_nodes in "${NODE_LIST[@]}"; do
    run_case "$num_nodes"
done

{
    echo "============================================================"
    echo " QMSum Node Sweep Summary"
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
