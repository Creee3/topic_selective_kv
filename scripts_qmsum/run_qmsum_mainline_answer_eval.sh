#!/bin/bash
# ============================================================================
# Current mainline answer evaluation for QMSum.
#
# Mainline:
#   lexical coarse routing
#   -> top1 topic
#   -> Q-K fine chunk routing
#   -> route_top_k=12
#   -> evaluate final answer quality and context saving
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_mainline_answer_eval.sh
#
# Optional overrides:
#   START_DOC=5 END_DOC=10 MAX_QUERIES=1 GPU_ID=2 \
#   bash scripts_qmsum/run_qmsum_mainline_answer_eval.sh
#
# Optional structure / safety overrides:
#   HIER_TOP_TOPICS=2 MAX_TOKENS=8000 \
#   bash scripts_qmsum/run_qmsum_mainline_answer_eval.sh
#
# Optional answer-optimization overrides:
#   DYNAMIC_ROUTE_BUDGET=1 ANSWER_EVIDENCE_ORDER=qk_then_time \
#   SELECTED_ANSWER_CONTEXT_MODE=chunk_turns ANSWER_PROMPT_STYLE=strict \
#   CASE_SUMMARY_TAG=answer_strict_chunkturns_5docs_q5 \
#   bash scripts_qmsum/run_qmsum_mainline_answer_eval.sh
# ============================================================================

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}
NUM_NODES=${NUM_NODES:-4}
GPU_ID=${GPU_ID:-0}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-10}
MAX_QUERIES=${MAX_QUERIES:-1}
CHUNK_SIZE=${CHUNK_SIZE:-128}
ROUTE_TOP_K=${ROUTE_TOP_K:-12}
ROUTE_NEIGHBOR_EXPAND=${ROUTE_NEIGHBOR_EXPAND:-0}
HIER_TOP_TOPICS=${HIER_TOP_TOPICS:-1}
MAX_TOKENS=${MAX_TOKENS:-0}
USE_PER_HEAD=${USE_PER_HEAD:-1}
ANSWER_MAX_NEW_TOKENS=${ANSWER_MAX_NEW_TOKENS:-96}
DYNAMIC_ROUTE_BUDGET=${DYNAMIC_ROUTE_BUDGET:-0}
DYNAMIC_SUMMARY_TOP_K=${DYNAMIC_SUMMARY_TOP_K:-16}
DYNAMIC_DETAIL_TOP_K=${DYNAMIC_DETAIL_TOP_K:-8}
DYNAMIC_BALANCED_TOP_K=${DYNAMIC_BALANCED_TOP_K:-12}
DYNAMIC_CANDIDATE_POOL_BUDGET=${DYNAMIC_CANDIDATE_POOL_BUDGET:-0}
DYNAMIC_CANDIDATE_POOL_BUDGET_MAP=${DYNAMIC_CANDIDATE_POOL_BUDGET_MAP:-summary:64,detail:48,balanced:48,default:48}
DYNAMIC_CANDIDATE_POOL_MIN_KEEP=${DYNAMIC_CANDIDATE_POOL_MIN_KEEP:-24}
ROUTE_CANDIDATE_PREFILTER=${ROUTE_CANDIDATE_PREFILTER:-none}
ROUTE_CANDIDATE_PREFILTER_FACTOR=${ROUTE_CANDIDATE_PREFILTER_FACTOR:-4}
ROUTE_CANDIDATE_PREFILTER_MIN_KEEP=${ROUTE_CANDIDATE_PREFILTER_MIN_KEEP:-24}
ROUTE_CANDIDATE_PREFILTER_MAX_KEEP=${ROUTE_CANDIDATE_PREFILTER_MAX_KEEP:-96}
ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=${ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO:-0.0}
ROUTE_COARSE_SEGMENT_GATE=${ROUTE_COARSE_SEGMENT_GATE:-none}
ROUTE_COARSE_SEGMENT_SIZE=${ROUTE_COARSE_SEGMENT_SIZE:-4}
ROUTE_COARSE_SEGMENT_KEEP_RATIO=${ROUTE_COARSE_SEGMENT_KEEP_RATIO:-0.5}
ROUTE_COARSE_SEGMENT_MIN_KEEP=${ROUTE_COARSE_SEGMENT_MIN_KEEP:-48}
ROUTE_COARSE_SEGMENT_MAX_KEEP=${ROUTE_COARSE_SEGMENT_MAX_KEEP:-0}
QK_SCORE_BATCH_SIZE=${QK_SCORE_BATCH_SIZE:-32}
CACHE_CANDIDATE_KEYS=${CACHE_CANDIDATE_KEYS:-0}
CACHE_QUERY_Q=${CACHE_QUERY_Q:-1}
SCORING_LAYERS=${SCORING_LAYERS:-}
ANSWER_EVIDENCE_ORDER=${ANSWER_EVIDENCE_ORDER:-time}
SELECTED_ANSWER_CONTEXT_MODE=${SELECTED_ANSWER_CONTEXT_MODE:-turns}
ANSWER_PROMPT_STYLE=${ANSWER_PROMPT_STYLE:-basic}
ANSWER_EVIDENCE_MAX_ENTRIES=${ANSWER_EVIDENCE_MAX_ENTRIES:-80}
ANSWER_EVIDENCE_MAX_CHARS=${ANSWER_EVIDENCE_MAX_CHARS:-600}
FETCH_BANDWIDTH_GBPS=${FETCH_BANDWIDTH_GBPS:-25.0}
PER_NODE_RTT_MS=${PER_NODE_RTT_MS:-1.0}
PER_SEGMENT_OVERHEAD_MS=${PER_SEGMENT_OVERHEAD_MS:-0.15}
DECODE_STARTUP_MS=${DECODE_STARTUP_MS:-15.0}
QUERY_TOKENIZER_WARMUP=${QUERY_TOKENIZER_WARMUP:-1}
CASE_SUMMARY_TAG=${CASE_SUMMARY_TAG:-mainline_lexical_top1_chunks_${ROUTE_TOP_K}_answers}
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}
LOG_DIR=${LOG_DIR:-logs/qmsum_mainline_answer_eval}
LOG_FILE="$LOG_DIR/mainline_answer_eval.log"

mkdir -p "$LOG_DIR"

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

if [ "$RESUME_IF_LOG_OK" -eq 1 ] && has_complete_log "$LOG_FILE"; then
    echo "Existing complete log detected. Reuse: $LOG_FILE"
    exit 0
fi

extra_args=()
if [ "$USE_PER_HEAD" -eq 1 ]; then
    extra_args+=(--route_per_head)
fi
if [ "$DYNAMIC_ROUTE_BUDGET" -eq 1 ]; then
    extra_args+=(--dynamic_route_budget)
fi
if [ "$DYNAMIC_CANDIDATE_POOL_BUDGET" -eq 1 ]; then
    extra_args+=(--dynamic_candidate_pool_budget)
fi
if [ "$CACHE_CANDIDATE_KEYS" -eq 1 ]; then
    extra_args+=(--cache_candidate_keys)
fi
if [ "$CACHE_QUERY_Q" -eq 1 ]; then
    extra_args+=(--cache_query_q)
else
    extra_args+=(--no_cache_query_q)
fi
if [ "$MAX_TOKENS" -gt 0 ]; then
    extra_args+=(--max_tokens "$MAX_TOKENS")
fi
if [ -n "$SCORING_LAYERS" ]; then
    extra_args+=(--scoring_layers "$SCORING_LAYERS")
fi

echo "============================================================"
echo " QMSum Mainline Answer Eval"
echo " GPU_ID=$GPU_ID"
echo " docs=$START_DOC:$END_DOC"
echo " NUM_NODES=$NUM_NODES"
echo " MAX_QUERIES=$MAX_QUERIES"
echo " HIER_TOP_TOPICS=$HIER_TOP_TOPICS"
echo " ROUTE_TOP_K=$ROUTE_TOP_K"
echo " ROUTE_NEIGHBOR_EXPAND=$ROUTE_NEIGHBOR_EXPAND"
echo " USE_PER_HEAD=$USE_PER_HEAD"
echo " MAX_TOKENS=$MAX_TOKENS"
echo " DYNAMIC_ROUTE_BUDGET=$DYNAMIC_ROUTE_BUDGET"
echo " DYNAMIC_SUMMARY_TOP_K=$DYNAMIC_SUMMARY_TOP_K"
echo " DYNAMIC_DETAIL_TOP_K=$DYNAMIC_DETAIL_TOP_K"
echo " DYNAMIC_BALANCED_TOP_K=$DYNAMIC_BALANCED_TOP_K"
echo " DYNAMIC_CANDIDATE_POOL_BUDGET=$DYNAMIC_CANDIDATE_POOL_BUDGET"
echo " DYNAMIC_CANDIDATE_POOL_BUDGET_MAP=$DYNAMIC_CANDIDATE_POOL_BUDGET_MAP"
echo " DYNAMIC_CANDIDATE_POOL_MIN_KEEP=$DYNAMIC_CANDIDATE_POOL_MIN_KEEP"
echo " ROUTE_CANDIDATE_PREFILTER=$ROUTE_CANDIDATE_PREFILTER"
echo " ROUTE_CANDIDATE_PREFILTER_FACTOR=$ROUTE_CANDIDATE_PREFILTER_FACTOR"
echo " ROUTE_CANDIDATE_PREFILTER_MIN_KEEP=$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP"
echo " ROUTE_CANDIDATE_PREFILTER_MAX_KEEP=$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP"
echo " ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO=$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO"
echo " ROUTE_COARSE_SEGMENT_GATE=$ROUTE_COARSE_SEGMENT_GATE"
echo " ROUTE_COARSE_SEGMENT_SIZE=$ROUTE_COARSE_SEGMENT_SIZE"
echo " ROUTE_COARSE_SEGMENT_KEEP_RATIO=$ROUTE_COARSE_SEGMENT_KEEP_RATIO"
echo " ROUTE_COARSE_SEGMENT_MIN_KEEP=$ROUTE_COARSE_SEGMENT_MIN_KEEP"
echo " ROUTE_COARSE_SEGMENT_MAX_KEEP=$ROUTE_COARSE_SEGMENT_MAX_KEEP"
echo " QK_SCORE_BATCH_SIZE=$QK_SCORE_BATCH_SIZE"
echo " CACHE_CANDIDATE_KEYS=$CACHE_CANDIDATE_KEYS"
echo " CACHE_QUERY_Q=$CACHE_QUERY_Q"
echo " SCORING_LAYERS=${SCORING_LAYERS:-default}"
echo " ANSWER_EVIDENCE_ORDER=$ANSWER_EVIDENCE_ORDER"
echo " SELECTED_ANSWER_CONTEXT_MODE=$SELECTED_ANSWER_CONTEXT_MODE"
echo " ANSWER_PROMPT_STYLE=$ANSWER_PROMPT_STYLE"
echo " ANSWER_MAX_NEW_TOKENS=$ANSWER_MAX_NEW_TOKENS"
echo " CASE_SUMMARY_TAG=$CASE_SUMMARY_TAG"
echo " FETCH_BANDWIDTH_GBPS=$FETCH_BANDWIDTH_GBPS"
echo " PER_NODE_RTT_MS=$PER_NODE_RTT_MS"
echo " PER_SEGMENT_OVERHEAD_MS=$PER_SEGMENT_OVERHEAD_MS"
echo " DECODE_STARTUP_MS=$DECODE_STARTUP_MS"
echo " QUERY_TOKENIZER_WARMUP=$QUERY_TOKENIZER_WARMUP"
echo " LOG_DIR=$LOG_DIR"
echo " LOG_FILE=$LOG_FILE"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python qmsum_mainline.py \
    --data_path "$DATA_PATH" \
    --model_id "$MODEL_ID" \
    --num_gpus "$NGPUS" \
    --max_gpu_memory "$MEM" \
    --num_nodes "$NUM_NODES" \
    --start_doc "$START_DOC" \
    --end_doc "$END_DOC" \
    --max_queries_per_doc "$MAX_QUERIES" \
    --routing_granularity hierarchical \
    --hier_top_topics "$HIER_TOP_TOPICS" \
    --hier_top_strategy lexical \
    --hier_topic_score_mode sum \
    --route_chunk_size "$CHUNK_SIZE" \
    --route_top_k "$ROUTE_TOP_K" \
    --route_neighbor_expand "$ROUTE_NEIGHBOR_EXPAND" \
    --dynamic_summary_top_k "$DYNAMIC_SUMMARY_TOP_K" \
    --dynamic_detail_top_k "$DYNAMIC_DETAIL_TOP_K" \
    --dynamic_balanced_top_k "$DYNAMIC_BALANCED_TOP_K" \
    --dynamic_candidate_pool_budget_map "$DYNAMIC_CANDIDATE_POOL_BUDGET_MAP" \
    --dynamic_candidate_pool_min_keep "$DYNAMIC_CANDIDATE_POOL_MIN_KEEP" \
    --route_candidate_prefilter "$ROUTE_CANDIDATE_PREFILTER" \
    --route_candidate_prefilter_factor "$ROUTE_CANDIDATE_PREFILTER_FACTOR" \
    --route_candidate_prefilter_min_keep "$ROUTE_CANDIDATE_PREFILTER_MIN_KEEP" \
    --route_candidate_prefilter_max_keep "$ROUTE_CANDIDATE_PREFILTER_MAX_KEEP" \
    --route_candidate_prefilter_min_prune_ratio "$ROUTE_CANDIDATE_PREFILTER_MIN_PRUNE_RATIO" \
    --route_coarse_segment_gate "$ROUTE_COARSE_SEGMENT_GATE" \
    --route_coarse_segment_size "$ROUTE_COARSE_SEGMENT_SIZE" \
    --route_coarse_segment_keep_ratio "$ROUTE_COARSE_SEGMENT_KEEP_RATIO" \
    --route_coarse_segment_min_keep "$ROUTE_COARSE_SEGMENT_MIN_KEEP" \
    --route_coarse_segment_max_keep "$ROUTE_COARSE_SEGMENT_MAX_KEEP" \
    --qk_score_batch_size "$QK_SCORE_BATCH_SIZE" \
    --answer_evidence_order "$ANSWER_EVIDENCE_ORDER" \
    --eval_answers \
    --answer_max_new_tokens "$ANSWER_MAX_NEW_TOKENS" \
    --selected_answer_context_mode "$SELECTED_ANSWER_CONTEXT_MODE" \
    --answer_prompt_style "$ANSWER_PROMPT_STYLE" \
    --answer_evidence_max_entries "$ANSWER_EVIDENCE_MAX_ENTRIES" \
    --answer_evidence_max_chars "$ANSWER_EVIDENCE_MAX_CHARS" \
    --fetch_bandwidth_gbps "$FETCH_BANDWIDTH_GBPS" \
    --per_node_rtt_ms "$PER_NODE_RTT_MS" \
    --per_segment_overhead_ms "$PER_SEGMENT_OVERHEAD_MS" \
    --decode_startup_ms "$DECODE_STARTUP_MS" \
    --query_tokenizer_warmup "$QUERY_TOKENIZER_WARMUP" \
    --case_summary_tag "$CASE_SUMMARY_TAG" \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG_FILE"
PIPE_STATUS=${PIPESTATUS[0]}
exit "$PIPE_STATUS"
