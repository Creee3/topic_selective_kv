#!/bin/bash
# ============================================================================
# QMSum selective mainline vs CacheGen-full estimated baseline.
#
# CacheGen-full here is an estimated baseline:
#   F1 proxy = full-context answer F1
#   TTFT = measured CacheGen full-KV compressed bytes + estimated transfer
# It does not yet decode CacheGen KV and regenerate answers.
# ============================================================================

set -u

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
GPU_ID=${GPU_ID:-0}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}

START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-3}
MAX_QUERIES=${MAX_QUERIES:-2}
NUM_NODES=${NUM_NODES:-4}
MAX_TOKENS=${MAX_TOKENS:-0}

EVAL_ANSWERS=${EVAL_ANSWERS:-1}
CACHEGEN_QUANT_LEVEL=${CACHEGEN_QUANT_LEVEL:-2}
CACHEGEN_CHUNK_SIZE=${CACHEGEN_CHUNK_SIZE:-256}
CACHEGEN_MODEL_NAME=${CACHEGEN_MODEL_NAME:-mistral-community/Mistral-7B-v0.2}
CACHEGEN_INCLUDE_ENCODE_TIME=${CACHEGEN_INCLUDE_ENCODE_TIME:-0}
CACHEGEN_DECODE_MS=${CACHEGEN_DECODE_MS:-0.0}
CACHEGEN_SEGMENT_COUNT_MODE=${CACHEGEN_SEGMENT_COUNT_MODE:-one}

CASE_SUMMARY_TAG=${CASE_SUMMARY_TAG:-cachegen_compare_q${MAX_QUERIES}}
LOG_DIR=${LOG_DIR:-logs/qmsum_cachegen_compare_${START_DOC}_${END_DOC}_q${MAX_QUERIES}}
LOG_FILE="$LOG_DIR/cachegen_compare.log"
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}

mkdir -p "$LOG_DIR"

has_complete_log() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 1
    fi
    grep -Fq "QMSum routing summary" "$file" || return 1
    grep -Fq "CacheGen-full estimated baseline" "$file" || return 1
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

echo "============================================================"
echo " QMSum CacheGen Compare"
echo " profile=current"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries=$MAX_QUERIES"
echo " gpu=$GPU_ID"
echo " num_nodes=$NUM_NODES"
echo " eval_answers=$EVAL_ANSWERS"
echo " cachegen_quant_level=$CACHEGEN_QUANT_LEVEL"
echo " cachegen_chunk_size=$CACHEGEN_CHUNK_SIZE"
echo " cachegen_include_encode_time=$CACHEGEN_INCLUDE_ENCODE_TIME"
echo " cachegen_segment_count_mode=$CACHEGEN_SEGMENT_COUNT_MODE"
echo " log=$LOG_FILE"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python qmsum_mainline.py \
    --mainline_profile current \
    --data_path "$DATA_PATH" \
    --model_id "$MODEL_ID" \
    --num_gpus "$NGPUS" \
    --max_gpu_memory "$MEM" \
    --num_nodes "$NUM_NODES" \
    --start_doc "$START_DOC" \
    --end_doc "$END_DOC" \
    --max_queries_per_doc "$MAX_QUERIES" \
    --case_summary_tag "$CASE_SUMMARY_TAG" \
    --eval_cachegen_full \
    --cachegen_model_name "$CACHEGEN_MODEL_NAME" \
    --cachegen_quant_level "$CACHEGEN_QUANT_LEVEL" \
    --cachegen_chunk_size "$CACHEGEN_CHUNK_SIZE" \
    --cachegen_include_encode_time "$CACHEGEN_INCLUDE_ENCODE_TIME" \
    --cachegen_decode_ms "$CACHEGEN_DECODE_MS" \
    --cachegen_segment_count_mode "$CACHEGEN_SEGMENT_COUNT_MODE" \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG_FILE"

exit "${PIPESTATUS[0]}"
