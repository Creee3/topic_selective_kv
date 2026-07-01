#!/bin/bash
# ============================================================================
# Clean current QMSum mainline runner.
#
# This is the entry point to use when you want the latest stabilized
# active-node v2 selective-fetch path, not an ablation branch.
#
# Current profile:
#   manually/dataset labeled topics as deployable semantic nodes
#   -> all topic/block candidates expose lightweight remote-node summaries
#   -> request node scores summaries with query-Q
#   -> rank-fusion combines summary/Q-K signals
#   -> selected full-KV blocks are fetched
#   -> virtual-node transfer accounting
#   -> optional answer F1
# ============================================================================

set -u

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
GPU_ID=${GPU_ID:-0}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}

START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-30}
MAX_QUERIES=${MAX_QUERIES:-5}
NUM_NODES=${NUM_NODES:-4}
MAX_TOKENS=${MAX_TOKENS:-0}
CACHE_QUERY_Q=${CACHE_QUERY_Q:-1}
QK_AGGREGATION=${QK_AGGREGATION:-}
QK_TOPK=${QK_TOPK:-}
QK_TOKEN_POOLING=${QK_TOKEN_POOLING:-}
QK_QUERY_TOPK_RATIO=${QK_QUERY_TOPK_RATIO:-}
MAINLINE_PROFILE=${MAINLINE_PROFILE:-current}
TTFT_MODEL=${TTFT_MODEL:-active_node_v2}

EVAL_ANSWERS=${EVAL_ANSWERS:-1}
EVAL_ORACLE_ANSWERS=${EVAL_ORACLE_ANSWERS:-1}
LIGHT_OUTPUT=${LIGHT_OUTPUT:-0}
WRITE_ANSWER_JSONL=${WRITE_ANSWER_JSONL:-1}
WRITE_ANSWER_MD=${WRITE_ANSWER_MD:-1}
CASE_SUMMARY_TAG=${CASE_SUMMARY_TAG:-current_mainline}
LOG_DIR=${LOG_DIR:-logs/qmsum_current_mainline_${START_DOC}_${END_DOC}_q${MAX_QUERIES}}
LOG_FILE=${LOG_FILE:-"$LOG_DIR/${MAINLINE_PROFILE}.log"}
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}

mkdir -p "$LOG_DIR"

has_complete_log() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 1
    fi
    grep -Fq "QMSum routing summary" "$file" || return 1
    grep -Fq "Saved to outputs/" "$file" || return 1
    return 0
}

if [ "$RESUME_IF_LOG_OK" -eq 1 ] && has_complete_log "$LOG_FILE"; then
    echo "Existing complete log detected. Reuse: $LOG_FILE"
    exit 0
fi

extra_args=()
if [ "$MAX_TOKENS" -gt 0 ]; then
    extra_args+=(--max_tokens "$MAX_TOKENS")
fi
if [ "$EVAL_ANSWERS" -eq 1 ]; then
    extra_args+=(--eval_answers)
fi
if [ "$EVAL_ORACLE_ANSWERS" -eq 0 ]; then
    extra_args+=(--no_eval_oracle_answers)
fi
if [ "$LIGHT_OUTPUT" -eq 1 ]; then
    extra_args+=(--light_output)
fi
if [ "$WRITE_ANSWER_JSONL" -eq 0 ]; then
    extra_args+=(--no_answer_jsonl)
fi
if [ "$WRITE_ANSWER_MD" -eq 0 ]; then
    extra_args+=(--no_answer_markdown)
fi
if [ "$CACHE_QUERY_Q" -eq 1 ]; then
    extra_args+=(--cache_query_q)
else
    extra_args+=(--no_cache_query_q)
fi
if [ -n "$QK_AGGREGATION" ]; then
    extra_args+=(--qk_aggregation "$QK_AGGREGATION")
fi
if [ -n "$QK_TOPK" ]; then
    extra_args+=(--qk_topk "$QK_TOPK")
fi
if [ -n "$QK_TOKEN_POOLING" ]; then
    extra_args+=(--qk_token_pooling "$QK_TOKEN_POOLING")
fi
if [ -n "$QK_QUERY_TOPK_RATIO" ]; then
    extra_args+=(--qk_query_topk_ratio "$QK_QUERY_TOPK_RATIO")
fi

echo "============================================================"
echo " QMSum Current Mainline"
echo " profile=$MAINLINE_PROFILE"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries=$MAX_QUERIES"
echo " gpu=$GPU_ID"
echo " num_nodes=$NUM_NODES"
echo " eval_answers=$EVAL_ANSWERS"
echo " eval_oracle_answers=$EVAL_ORACLE_ANSWERS"
echo " light_output=$LIGHT_OUTPUT"
echo " write_answer_jsonl=$WRITE_ANSWER_JSONL"
echo " write_answer_md=$WRITE_ANSWER_MD"
echo " cache_query_q=$CACHE_QUERY_Q"
echo " ttft_model=$TTFT_MODEL"
echo " qk_aggregation=$QK_AGGREGATION"
echo " qk_topk=$QK_TOPK"
echo " qk_token_pooling=$QK_TOKEN_POOLING"
echo " qk_query_topk_ratio=$QK_QUERY_TOPK_RATIO"
echo " log=$LOG_FILE"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python qmsum_mainline.py \
    --mainline_profile "$MAINLINE_PROFILE" \
    --ttft_model "$TTFT_MODEL" \
    --data_path "$DATA_PATH" \
    --model_id "$MODEL_ID" \
    --num_gpus "$NGPUS" \
    --max_gpu_memory "$MEM" \
    --num_nodes "$NUM_NODES" \
    --start_doc "$START_DOC" \
    --end_doc "$END_DOC" \
    --max_queries_per_doc "$MAX_QUERIES" \
    --case_summary_tag "$CASE_SUMMARY_TAG" \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG_FILE"

exit "${PIPESTATUS[0]}"
