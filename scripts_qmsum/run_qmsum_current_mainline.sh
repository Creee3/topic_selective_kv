#!/bin/bash
# ============================================================================
# Clean current QMSum mainline runner.
#
# This is the entry point to use when you want the latest stabilized
# selective-fetch path, not an ablation branch.
#
# Current profile:
#   manually/dataset labeled topics as deployable semantic nodes
#   -> lexical top-1 topic routing
#   -> lexical candidate prefilter
#   -> lexical coarse segment gate
#   -> batched exact Q-K chunk scoring
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
MAINLINE_PROFILE=${MAINLINE_PROFILE:-current}

EVAL_ANSWERS=${EVAL_ANSWERS:-1}
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
if [ "$CACHE_QUERY_Q" -eq 1 ]; then
    extra_args+=(--cache_query_q)
else
    extra_args+=(--no_cache_query_q)
fi

echo "============================================================"
echo " QMSum Current Mainline"
echo " profile=$MAINLINE_PROFILE"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries=$MAX_QUERIES"
echo " gpu=$GPU_ID"
echo " num_nodes=$NUM_NODES"
echo " eval_answers=$EVAL_ANSWERS"
echo " cache_query_q=$CACHE_QUERY_Q"
echo " log=$LOG_FILE"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python qmsum_mainline.py \
    --mainline_profile "$MAINLINE_PROFILE" \
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
