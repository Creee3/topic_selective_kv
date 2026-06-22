#!/bin/bash
# ============================================================================
# One-click stability check for the current best ShareGPT chunk-routing setup.
#
# Purpose:
#   Run several ShareGPT doc ranges with the current best configuration and
#   summarize whether the selected_count conclusion is stable across slices.
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash run_sharegpt_stability_check.sh
#
# Outputs:
#   logs/sharegpt_stability_check/summary.tsv
#   logs/sharegpt_stability_check/summary.txt
# ============================================================================

MODEL_ID=~/models/mistral-7b/
NGPUS=1
MEM=40
NUM_NODES=4
CHUNK_SIZE=128
ROUTE_TOP_K=4
LOG_DIR=logs/sharegpt_stability_check
SUMMARY_TSV="$LOG_DIR/summary.tsv"
SUMMARY_TXT="$LOG_DIR/summary.txt"

RANGES=(
  "0 60"
  "60 120"
  "120 180"
)

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " ShareGPT Stability Check"
echo " model=$MODEL_ID"
echo " nodes=$NUM_NODES"
echo " chunk_size=$CHUNK_SIZE"
echo " route_top_k=$ROUTE_TOP_K"
echo " route_per_head=1"
echo " chunk_node_score_mode=selected_count"
echo " note=Ranges are chosen for the 200-sample ShareGPT test file"
echo " logs=$LOG_DIR"
echo "============================================================"

cat > "$SUMMARY_TSV" <<EOF
case	start_doc	end_doc	qk_top1_pct	qk_top2_pct	selected_node_pct	selected_turn_pct	selected_first_chunk_pct	qk_avg_var	qk_avg_range	distinguishable_pct
EOF

extract_line() {
    local pattern="$1"
    local file="$2"
    grep -F "$pattern" "$file" | tail -n 1
}

append_summary() {
    local case_name="$1"
    local start_doc="$2"
    local end_doc="$3"
    local log_file="$4"

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
        "$case_name" "$start_doc" "$end_doc" "$qk_top1" "$qk_top2" \
        "$selected_node" "$selected_turn" "$selected_first_chunk" \
        "$qk_var" "$qk_range" "$distinguishable" >> "$SUMMARY_TSV"
}

run_case() {
    local start_doc="$1"
    local end_doc="$2"
    local case_name="docs_${start_doc}_${end_doc}"
    local log_file="$LOG_DIR/${case_name}.log"

    echo ""
    echo "============================================================"
    echo " Running case: $case_name"
    echo " Log file: $log_file"
    echo "============================================================"

    python distributed_sim.py \
        --model_id "$MODEL_ID" \
        --num_gpus "$NGPUS" \
        --max_gpu_memory "$MEM" \
        --num_nodes "$NUM_NODES" \
        --start_doc "$start_doc" \
        --end_doc "$end_doc" \
        --passkey \
        --baselines \
        --routing_granularity chunk \
        --route_chunk_size "$CHUNK_SIZE" \
        --route_top_k "$ROUTE_TOP_K" \
        --chunk_node_score_mode selected_count \
        --route_per_head \
        2>&1 | tee "$log_file"

    append_summary "$case_name" "$start_doc" "$end_doc" "$log_file"
}

for range in "${RANGES[@]}"; do
    run_case ${range}
done

{
    echo "============================================================"
    echo " ShareGPT Stability Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Stability check complete"
echo " Check logs under: $LOG_DIR"
echo " Key summary: $SUMMARY_TXT"
echo "============================================================"
