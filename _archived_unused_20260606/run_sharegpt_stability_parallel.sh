#!/bin/bash
# ============================================================================
# Parallel ShareGPT stability check for the current best chunk-routing setup.
#
# Purpose:
#   Run several ShareGPT ranges in parallel on different GPUs, then summarize
#   the key routing metrics after all jobs finish.
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash run_sharegpt_stability_parallel.sh
#
# Optional:
#   GPUS="1 2 5" bash run_sharegpt_stability_parallel.sh
#
# Outputs:
#   logs/sharegpt_stability_parallel/summary.tsv
#   logs/sharegpt_stability_parallel/summary.txt
# ============================================================================

MODEL_ID=~/models/mistral-7b/
NGPUS=1
MEM=40
NUM_NODES=4
CHUNK_SIZE=128
ROUTE_TOP_K=4
LOG_DIR=logs/sharegpt_stability_parallel
SUMMARY_TSV="$LOG_DIR/summary.tsv"
SUMMARY_TXT="$LOG_DIR/summary.txt"

RANGES=(
  "0 60"
  "60 120"
  "120 180"
)

DEFAULT_GPUS=(1 2 5)
if [ -n "${GPUS:-}" ]; then
    read -r -a GPU_LIST <<< "$GPUS"
else
    GPU_LIST=("${DEFAULT_GPUS[@]}")
fi

mkdir -p "$LOG_DIR"

if [ "${#GPU_LIST[@]}" -lt "${#RANGES[@]}" ]; then
    echo "Need at least ${#RANGES[@]} GPUs, but only got ${#GPU_LIST[@]}."
    echo "Example: GPUS=\"1 2 5\" bash run_sharegpt_stability_parallel.sh"
    exit 1
fi

echo "============================================================"
echo " Parallel ShareGPT Stability Check"
echo " model=$MODEL_ID"
echo " nodes=$NUM_NODES"
echo " chunk_size=$CHUNK_SIZE"
echo " route_top_k=$ROUTE_TOP_K"
echo " route_per_head=1"
echo " chunk_node_score_mode=selected_count"
echo " note=Ranges are chosen for the 200-sample ShareGPT test file"
echo " gpus=${GPU_LIST[*]}"
echo " logs=$LOG_DIR"
echo "============================================================"

extract_line() {
    local pattern="$1"
    local file="$2"
    grep -F "$pattern" "$file" | tail -n 1
}

append_summary() {
    local case_name="$1"
    local gpu_id="$2"
    local start_doc="$3"
    local end_doc="$4"
    local log_file="$5"

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

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$case_name" "$gpu_id" "$start_doc" "$end_doc" "$qk_top1" "$qk_top2" \
        "$selected_node" "$selected_turn" "$selected_first_chunk" \
        "$qk_var" "$qk_range" "$distinguishable" >> "$SUMMARY_TSV"
}

run_case_async() {
    local gpu_id="$1"
    local start_doc="$2"
    local end_doc="$3"
    local case_name="docs_${start_doc}_${end_doc}"
    local log_file="$LOG_DIR/${case_name}.log"

    echo ""
    echo "============================================================"
    echo " Launching case: $case_name"
    echo " GPU: $gpu_id"
    echo " Log file: $log_file"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$gpu_id" \
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
        > "$log_file" 2>&1 &

    echo $!
}

declare -a PIDS
declare -a CASES
declare -a CASE_GPUS
declare -a CASE_STARTS
declare -a CASE_ENDS

for idx in "${!RANGES[@]}"; do
    read -r start_doc end_doc <<< "${RANGES[$idx]}"
    gpu_id="${GPU_LIST[$idx]}"
    pid=$(run_case_async "$gpu_id" "$start_doc" "$end_doc" | tail -n 1)

    PIDS+=("$pid")
    CASES+=("docs_${start_doc}_${end_doc}")
    CASE_GPUS+=("$gpu_id")
    CASE_STARTS+=("$start_doc")
    CASE_ENDS+=("$end_doc")
done

echo ""
echo "Waiting for all jobs to finish..."

failure_count=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    case_name="${CASES[$idx]}"
    if wait "$pid"; then
        echo "  [OK]   $case_name (pid=$pid)"
    else
        echo "  [FAIL] $case_name (pid=$pid)"
        failure_count=$((failure_count + 1))
    fi
done

cat > "$SUMMARY_TSV" <<EOF
case	gpu	start_doc	end_doc	qk_top1_pct	qk_top2_pct	selected_node_pct	selected_turn_pct	selected_first_chunk_pct	qk_avg_var	qk_avg_range	distinguishable_pct
EOF

for idx in "${!CASES[@]}"; do
    case_name="${CASES[$idx]}"
    gpu_id="${CASE_GPUS[$idx]}"
    start_doc="${CASE_STARTS[$idx]}"
    end_doc="${CASE_ENDS[$idx]}"
    log_file="$LOG_DIR/${case_name}.log"

    if [ -f "$log_file" ]; then
        append_summary "$case_name" "$gpu_id" "$start_doc" "$end_doc" "$log_file"
    fi
done

{
    echo "============================================================"
    echo " Parallel ShareGPT Stability Summary"
    echo "============================================================"
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "$SUMMARY_TSV"
    else
        cat "$SUMMARY_TSV"
    fi
    echo ""
    echo "Failures: $failure_count"
} | tee "$SUMMARY_TXT"

echo ""
echo "============================================================"
echo " Parallel stability check complete"
echo " Check logs under: $LOG_DIR"
echo " Key summary: $SUMMARY_TXT"
echo "============================================================"
