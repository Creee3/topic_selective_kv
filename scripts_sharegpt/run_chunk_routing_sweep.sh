#!/bin/bash
# ============================================================================
# One-click sweep for chunk-level distributed KV routing on ShareGPT.
#
# Purpose:
#   Run the current best chunk-routing setting and compare the node-score
#   aggregation modes that map selected chunks back to final node ranking.
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash run_chunk_routing_sweep.sh
#
# Notes:
#   - This script focuses on the current diagnosis point:
#       chunk evidence -> final node ranking
#   - Logs are saved under logs/chunk_node_score_sweep/
#   - A compact summary is written to:
#       logs/chunk_node_score_sweep/summary.tsv
#       logs/chunk_node_score_sweep/summary.txt
# ============================================================================

MODEL_ID=~/models/mistral-7b/
NGPUS=1
MEM=40
NUM_NODES=4
START_DOC=0
END_DOC=200
CHUNK_SIZE=128
ROUTE_TOP_K=4
USE_PER_HEAD=1
LOG_DIR=logs/chunk_node_score_sweep
SUMMARY_TSV="$LOG_DIR/summary.tsv"
SUMMARY_TXT="$LOG_DIR/summary.txt"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Chunk-Level Node-Score Sweep"
echo " model=$MODEL_ID"
echo " docs=$START_DOC:$END_DOC"
echo " nodes=$NUM_NODES"
echo " chunk_size=$CHUNK_SIZE"
echo " route_top_k=$ROUTE_TOP_K"
echo " route_per_head=$USE_PER_HEAD"
echo " logs=$LOG_DIR"
echo "============================================================"

cat > "$SUMMARY_TSV" <<EOF
case	mode	qk_top1_pct	qk_top2_pct	selected_node_pct	selected_turn_pct	selected_first_chunk_pct	qk_avg_var	qk_avg_range	distinguishable_pct
EOF

extract_line() {
    local pattern="$1"
    local file="$2"
    grep -F "$pattern" "$file" | tail -n 1
}

append_summary() {
    local case_name="$1"
    local mode="$2"
    local log_file="$3"

    local qk_line selected_node_line selected_turn_line selected_first_chunk_line
    local var_line range_line dist_line
    local qk_top1 qk_top2 selected_node selected_turn selected_first_chunk
    local qk_var qk_range distinguishable

    qk_line=$(extract_line "qk top-1:" "$log_file")
    selected_node_line=$(extract_line "qk selected-node:" "$log_file")
    selected_turn_line=$(extract_line "qk selected-turn:" "$log_file")
    selected_first_chunk_line=$(extract_line "qk selected-first-chunk:" "$log_file")
    var_line=$(extract_line "Q-K avg score variance:" "$log_file")
    range_line=$(extract_line "Q-K avg score range:" "$log_file")
    dist_line=$(extract_line "distinguishable:" "$log_file")

    qk_top1=$(echo "$qk_line" | sed -E 's/.*\(([0-9.]+)%\), top-2: .*/\1/')
    qk_top2=$(echo "$qk_line" | sed -E 's/.*top-2: [0-9]+\/[0-9]+ \(([0-9.]+)%\).*/\1/')
    selected_node=$(echo "$selected_node_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    selected_turn=$(echo "$selected_turn_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    selected_first_chunk=$(echo "$selected_first_chunk_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    qk_var=$(echo "$var_line" | awk '{print $5}')
    qk_range=$(echo "$range_line" | awk '{print $5}')
    distinguishable=$(echo "$dist_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$case_name" "$mode" "$qk_top1" "$qk_top2" "$selected_node" \
        "$selected_turn" "$selected_first_chunk" "$qk_var" "$qk_range" \
        "$distinguishable" >> "$SUMMARY_TSV"
}

run_case() {
    local name="$1"
    local mode="$2"
    local extra_args=()

    local log_file="$LOG_DIR/${name}.log"
    echo ""
    echo "============================================================"
    echo " Running case: $name"
    echo " mode: $mode"
    echo " Log file: $log_file"
    echo "============================================================"

    if [ "$USE_PER_HEAD" -eq 1 ]; then
        extra_args+=(--route_per_head)
    fi

    python distributed_sim.py \
        --model_id "$MODEL_ID" \
        --num_gpus "$NGPUS" \
        --max_gpu_memory "$MEM" \
        --num_nodes "$NUM_NODES" \
        --start_doc "$START_DOC" \
        --end_doc "$END_DOC" \
        --passkey \
        --baselines \
        --routing_granularity chunk \
        --route_chunk_size "$CHUNK_SIZE" \
        --route_top_k "$ROUTE_TOP_K" \
        --chunk_node_score_mode "$mode" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"

    append_summary "$name" "$mode" "$log_file"
}

run_case mode_selected_sum selected_sum
run_case mode_selected_count selected_count
run_case mode_selected_max selected_max
run_case mode_all_chunk_max all_chunk_max

{
    echo "============================================================"
    echo " Chunk Node-Score Sweep Summary"
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
