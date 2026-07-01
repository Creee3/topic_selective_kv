#!/bin/bash
# ============================================================================
# QMSum CacheGen roundtrip answer smoke.
#
# This is the real CacheGen quality path:
#   full answer prompt KV -> CacheGen compress -> CacheGen decompress
#   -> generate from decoded KV -> answer F1
#
# Keep the default tiny. CacheGen roundtrip is much heavier than the previous
# estimated baseline because it compresses/decompresses one answer prompt per
# evaluated query.
# ============================================================================

set -u

MODEL_ID=${MODEL_ID:-~/models/mistral-7b/}
DATA_PATH=${DATA_PATH:-~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl}
GPU_ID=${GPU_ID:-0}
NGPUS=${NGPUS:-1}
MEM=${MEM:-40}

START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-2}
MAX_QUERIES=${MAX_QUERIES:-1}
NUM_NODES=${NUM_NODES:-4}
MAX_TOKENS=${MAX_TOKENS:-0}
MAINLINE_PROFILE=${MAINLINE_PROFILE:-current}
TTFT_MODEL=${TTFT_MODEL:-active_node_v2}

CACHEGEN_QUANT_LEVEL=${CACHEGEN_QUANT_LEVEL:-2}
CACHEGEN_CHUNK_SIZE=${CACHEGEN_CHUNK_SIZE:-256}
CACHEGEN_MODEL_NAME=${CACHEGEN_MODEL_NAME:-mistral-community/Mistral-7B-v0.2}
CACHEGEN_INCLUDE_ENCODE_TIME=${CACHEGEN_INCLUDE_ENCODE_TIME:-0}
CACHEGEN_SEGMENT_COUNT_MODE=${CACHEGEN_SEGMENT_COUNT_MODE:-cachegen_chunks}

CASE_SUMMARY_TAG=${CASE_SUMMARY_TAG:-cachegen_roundtrip_smoke_q${MAX_QUERIES}}
LOG_DIR=${LOG_DIR:-logs/qmsum_cachegen_roundtrip_smoke_${START_DOC}_${END_DOC}_q${MAX_QUERIES}}
LOG_FILE="$LOG_DIR/cachegen_roundtrip_smoke.log"

mkdir -p "$LOG_DIR"

extra_args=()
if [ "$MAX_TOKENS" -gt 0 ]; then
    extra_args+=(--max_tokens "$MAX_TOKENS")
fi

echo "============================================================"
echo " QMSum CacheGen Roundtrip Answer Smoke"
echo " profile=$MAINLINE_PROFILE"
echo " ttft_model=$TTFT_MODEL"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries=$MAX_QUERIES"
echo " gpu=$GPU_ID"
echo " cachegen_quant_level=$CACHEGEN_QUANT_LEVEL"
echo " cachegen_chunk_size=$CACHEGEN_CHUNK_SIZE"
echo " cachegen_segment_count_mode=$CACHEGEN_SEGMENT_COUNT_MODE"
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
    --eval_answers \
    --no_eval_oracle_answers \
    --eval_cachegen_full \
    --eval_cachegen_roundtrip_answer \
    --cachegen_model_name "$CACHEGEN_MODEL_NAME" \
    --cachegen_quant_level "$CACHEGEN_QUANT_LEVEL" \
    --cachegen_chunk_size "$CACHEGEN_CHUNK_SIZE" \
    --cachegen_include_encode_time "$CACHEGEN_INCLUDE_ENCODE_TIME" \
    --cachegen_segment_count_mode "$CACHEGEN_SEGMENT_COUNT_MODE" \
    --light_output \
    --no_answer_markdown \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG_FILE"

exit "${PIPESTATUS[0]}"
