#!/bin/bash
# ============================================================================
# Compare precomputed-topic embedding against candidate-turn rerank.
#
# Purpose:
#   Keep the coarse router cheap:
#     query -> precomputed topic representations -> top candidate topics
#
#   Then test whether a lightweight rerank inside those candidates can push the
#   correct topic from top-2 to top-1, without returning to global full-turn
#   scanning.
#
# Cases:
#   1. embedding
#   2. rerank (candidate_turns)
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_precomputed_rerank_compare.sh
#
# Optional overrides:
#   START_DOC=5 END_DOC=6 MAX_QUERIES=1 \
#   RERANK_CANDIDATE_TOPICS=2 HIER_TOP_TOPICS=1 \
#   GPU_EMBED=6 GPU_RERANK=7 PARALLEL_RUN=0 \
#   bash scripts_qmsum/run_qmsum_precomputed_rerank_compare.sh
#
# Outputs:
#   logs/qmsum_precomputed_rerank_compare/*.log
#   logs/qmsum_precomputed_rerank_compare/summary.tsv
#   logs/qmsum_precomputed_rerank_compare/summary.txt
# ============================================================================

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-6}
MAX_QUERIES=${MAX_QUERIES:-1}
HIER_TOP_TOPICS=${HIER_TOP_TOPICS:-1}
HIER_TOPIC_SCORE_MODE=${HIER_TOPIC_SCORE_MODE:-sum}
CHUNK_SIZE=${CHUNK_SIZE:-128}
ROUTE_TOP_K=${ROUTE_TOP_K:-16}
USE_PER_HEAD=${USE_PER_HEAD:-1}
EVAL_ANSWERS=${EVAL_ANSWERS:-1}
ANSWER_MAX_NEW_TOKENS=${ANSWER_MAX_NEW_TOKENS:-96}
TOPIC_PROTOTYPE_TURNS=${TOPIC_PROTOTYPE_TURNS:-5}
TOPIC_REPR_TEMPLATE=${TOPIC_REPR_TEMPLATE:-basic}
RERANK_CANDIDATE_TOPICS=${RERANK_CANDIDATE_TOPICS:-2}
RERANK_EMBEDDING_WEIGHT=${RERANK_EMBEDDING_WEIGHT:-1.0}
RERANK_QK_WEIGHT=${RERANK_QK_WEIGHT:-0.7}
GPU_EMBED=${GPU_EMBED:-6}
GPU_RERANK=${GPU_RERANK:-7}
PARALLEL_RUN=${PARALLEL_RUN:-0}
LOG_DIR=${LOG_DIR:-logs/qmsum_precomputed_rerank_compare}
SUMMARY_TSV="$LOG_DIR/summary.tsv"
SUMMARY_TXT="$LOG_DIR/summary.txt"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " QMSum Precomputed-Rerank Compare"
echo " model=$MODEL_ID"
echo " data=$DATA_PATH"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries_per_doc=$MAX_QUERIES"
echo " hier_top_topics=$HIER_TOP_TOPICS"
echo " hier_topic_score_mode=$HIER_TOPIC_SCORE_MODE"
echo " chunk_size=$CHUNK_SIZE"
echo " route_top_k=$ROUTE_TOP_K"
echo " route_per_head=$USE_PER_HEAD"
echo " eval_answers=$EVAL_ANSWERS"
echo " answer_max_new_tokens=$ANSWER_MAX_NEW_TOKENS"
echo " topic_prototype_turns=$TOPIC_PROTOTYPE_TURNS"
echo " topic_repr_template=$TOPIC_REPR_TEMPLATE"
echo " rerank_candidate_topics=$RERANK_CANDIDATE_TOPICS"
echo " rerank_embedding_weight=$RERANK_EMBEDDING_WEIGHT"
echo " rerank_qk_weight=$RERANK_QK_WEIGHT"
echo " gpu_embed=$GPU_EMBED"
echo " gpu_rerank=$GPU_RERANK"
echo " parallel_run=$PARALLEL_RUN"
echo " logs=$LOG_DIR"
echo "============================================================"

cat > "$SUMMARY_TSV" <<EOF
strategy	status	top1_pct	top2_pct	selected_topic_hit_pct	selected_turn_hit_pct	avg_turn_recall_pct	avg_turn_precision_pct	avg_turn_f1_pct	full_answer_f1_pct	selected_answer_f1_pct	answer_f1_delta_pct	ctx_token_saving_pct
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
    local strategy="$1"
    local status="$2"
    local log_file="$3"

    if [ "$status" != "OK" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$strategy" "$status" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    if ! grep -Fq "QMSum routing summary" "$log_file"; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$strategy" "PARSE_FAIL" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    local strategy_line selected_topic_line selected_turn_line
    local recall_line precision_line f1_line
    local full_answer_line selected_answer_line answer_delta_line ctx_saving_line

    strategy_line=$(extract_strategy_line "  Relevant-span topic hit rate \\(loose\\):" "${strategy} top-1:" "$log_file")
    selected_topic_line=$(extract_line "selected-topic hit (${strategy}):" "$log_file")
    selected_turn_line=$(extract_line "selected-turn hit:" "$log_file")
    recall_line=$(extract_line "avg turn recall:" "$log_file")
    precision_line=$(extract_line "avg turn precision:" "$log_file")
    f1_line=$(extract_line "avg turn F1:" "$log_file")
    full_answer_line=$(extract_line "avg full-answer F1:" "$log_file")
    selected_answer_line=$(extract_line "avg selective-answer F1:" "$log_file")
    answer_delta_line=$(extract_line "avg F1 delta:" "$log_file")
    ctx_saving_line=$(extract_line "avg ctx token saving:" "$log_file")

    if [ -z "$strategy_line" ] || [ -z "$selected_topic_line" ] || [ -z "$selected_turn_line" ] || \
       [ -z "$recall_line" ] || [ -z "$precision_line" ] || [ -z "$f1_line" ] || \
       [ -z "$full_answer_line" ] || [ -z "$selected_answer_line" ] || \
       [ -z "$answer_delta_line" ] || [ -z "$ctx_saving_line" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$strategy" "PARSE_FAIL" \
            "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" >> "$SUMMARY_TSV"
        return
    fi

    local top1_pct top2_pct selected_topic_pct selected_turn_pct
    local recall_pct precision_pct f1_pct
    local full_answer_pct selected_answer_pct answer_delta_pct ctx_saving_pct

    top1_pct=$(echo "$strategy_line" | sed -E 's/.*\(([0-9.]+)%\), top-2: .*/\1/')
    top2_pct=$(echo "$strategy_line" | sed -E 's/.*top-2: [0-9]+\/[0-9]+ \(([0-9.]+)%\).*/\1/')
    selected_topic_pct=$(echo "$selected_topic_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    selected_turn_pct=$(echo "$selected_turn_line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    recall_pct=$(echo "$recall_line" | awk '{print $4}' | tr -d '%')
    precision_pct=$(echo "$precision_line" | awk '{print $4}' | tr -d '%')
    f1_pct=$(echo "$f1_line" | awk '{print $4}' | tr -d '%')
    full_answer_pct=$(echo "$full_answer_line" | awk '{print $4}' | tr -d '%')
    selected_answer_pct=$(echo "$selected_answer_line" | awk '{print $4}' | tr -d '%')
    answer_delta_pct=$(echo "$answer_delta_line" | awk '{print $4}' | tr -d '%')
    ctx_saving_pct=$(echo "$ctx_saving_line" | awk '{print $5}' | tr -d '%')

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$strategy" "OK" \
        "$top1_pct" "$top2_pct" "$selected_topic_pct" "$selected_turn_pct" \
        "$recall_pct" "$precision_pct" "$f1_pct" \
        "$full_answer_pct" "$selected_answer_pct" "$answer_delta_pct" "$ctx_saving_pct" >> "$SUMMARY_TSV"
}

run_case() {
    local strategy="$1"
    local gpu_id="$2"
    local log_file="$LOG_DIR/${strategy}.log"
    local extra_args=()

    echo ""
    echo "============================================================"
    echo " Running strategy: $strategy"
    echo " GPU: $gpu_id"
    echo " Log file: $log_file"
    echo "============================================================"

    if [ "$USE_PER_HEAD" -eq 1 ]; then
        extra_args+=(--route_per_head)
    fi
    if [ "$EVAL_ANSWERS" -eq 1 ]; then
        extra_args+=(--eval_answers --answer_max_new_tokens "$ANSWER_MAX_NEW_TOKENS")
    fi
    if [ "$strategy" = "rerank" ]; then
        extra_args+=(
            --rerank_candidate_topics "$RERANK_CANDIDATE_TOPICS"
            --rerank_source candidate_turns
            --rerank_embedding_weight "$RERANK_EMBEDDING_WEIGHT"
            --rerank_qk_weight "$RERANK_QK_WEIGHT"
        )
    fi

    CUDA_VISIBLE_DEVICES="$gpu_id" python qmsum_sim.py \
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
        --hier_top_strategy "$strategy" \
        --hier_topic_score_mode "$HIER_TOPIC_SCORE_MODE" \
        --topic_embedding_source precomputed_topic_text \
        --topic_prototype_turns "$TOPIC_PROTOTYPE_TURNS" \
        --topic_representation_template "$TOPIC_REPR_TEMPLATE" \
        --route_chunk_size "$CHUNK_SIZE" \
        --route_top_k "$ROUTE_TOP_K" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    local pipe_status=${PIPESTATUS[0]}

    if [ "$pipe_status" -ne 0 ]; then
        echo " Strategy failed with exit code: $pipe_status" | tee -a "$log_file"
        append_summary "$strategy" "RUN_FAIL" "$log_file"
        return
    fi

    append_summary "$strategy" "OK" "$log_file"
}

if [ "$PARALLEL_RUN" -eq 1 ]; then
    run_case "embedding" "$GPU_EMBED" &
    pid_embed=$!
    run_case "rerank" "$GPU_RERANK" &
    pid_rerank=$!
    wait "$pid_embed"
    wait "$pid_rerank"
else
    run_case "embedding" "$GPU_EMBED"
    run_case "rerank" "$GPU_RERANK"
fi

{
    echo "============================================================"
    echo " QMSum Precomputed-Rerank Compare Summary"
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
