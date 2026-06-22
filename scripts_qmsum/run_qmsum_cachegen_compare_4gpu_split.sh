#!/bin/bash
# ============================================================================
# Run QMSum selective mainline vs CacheGen-full estimated baseline on 4 GPUs.
#
# Default split for docs 0:30:
#   GPU 0 -> docs 0:8
#   GPU 1 -> docs 8:16
#   GPU 2 -> docs 16:23
#   GPU 3 -> docs 23:30
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_cachegen_compare_4gpu_split.sh
# ============================================================================

set -u

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [ "$SCRIPT_DIR" = "$SCRIPT_PATH" ]; then
    SCRIPT_DIR="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"

START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-30}
MAX_QUERIES=${MAX_QUERIES:-5}
GPUS=${GPUS:-"0 1 2 3"}
NUM_NODES=${NUM_NODES:-4}
EVAL_ANSWERS=${EVAL_ANSWERS:-1}
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}

CACHEGEN_QUANT_LEVEL=${CACHEGEN_QUANT_LEVEL:-2}
CACHEGEN_CHUNK_SIZE=${CACHEGEN_CHUNK_SIZE:-256}
CACHEGEN_MODEL_NAME=${CACHEGEN_MODEL_NAME:-mistral-community/Mistral-7B-v0.2}
CACHEGEN_INCLUDE_ENCODE_TIME=${CACHEGEN_INCLUDE_ENCODE_TIME:-0}
CACHEGEN_DECODE_MS=${CACHEGEN_DECODE_MS:-0.0}
CACHEGEN_SEGMENT_COUNT_MODE=${CACHEGEN_SEGMENT_COUNT_MODE:-one}

LOG_ROOT=${LOG_ROOT:-logs/qmsum_cachegen_compare_split_${START_DOC}_${END_DOC}_q${MAX_QUERIES}}
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

read -r -a GPU_LIST <<< "$GPUS"
GPU_COUNT=${#GPU_LIST[@]}

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPUs configured. Set GPUS=\"0 1 2 3\" or similar."
    exit 1
fi

total_docs=$((END_DOC - START_DOC))
if [ "$total_docs" -le 0 ]; then
    echo "END_DOC must be greater than START_DOC"
    exit 1
fi

pids=()
labels=()
statuses=()

echo "============================================================"
echo " QMSum CacheGen Compare Split Run"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries=$MAX_QUERIES"
echo " gpus=$GPUS"
echo " num_nodes=$NUM_NODES"
echo " eval_answers=$EVAL_ANSWERS"
echo " cachegen_quant_level=$CACHEGEN_QUANT_LEVEL"
echo " cachegen_chunk_size=$CACHEGEN_CHUNK_SIZE"
echo " cachegen_include_encode_time=$CACHEGEN_INCLUDE_ENCODE_TIME"
echo " cachegen_segment_count_mode=$CACHEGEN_SEGMENT_COUNT_MODE"
echo " log_root=$LOG_ROOT"
echo "============================================================"

for i in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$i]}"
    doc_start=$((START_DOC + (total_docs * i) / GPU_COUNT))
    doc_end=$((START_DOC + (total_docs * (i + 1)) / GPU_COUNT))
    if [ "$doc_start" -ge "$doc_end" ]; then
        continue
    fi

    label="docs_${doc_start}_${doc_end}_gpu${gpu}"
    tag="cachegen_compare_${doc_start}_${doc_end}_q${MAX_QUERIES}"
    log_dir="$LOG_ROOT/$label"

    echo "Launch $label -> CASE_SUMMARY_TAG=$tag"
    (
        GPU_ID="$gpu" \
        START_DOC="$doc_start" \
        END_DOC="$doc_end" \
        MAX_QUERIES="$MAX_QUERIES" \
        NUM_NODES="$NUM_NODES" \
        EVAL_ANSWERS="$EVAL_ANSWERS" \
        CASE_SUMMARY_TAG="$tag" \
        LOG_DIR="$log_dir" \
        RESUME_IF_LOG_OK="$RESUME_IF_LOG_OK" \
        CACHEGEN_QUANT_LEVEL="$CACHEGEN_QUANT_LEVEL" \
        CACHEGEN_CHUNK_SIZE="$CACHEGEN_CHUNK_SIZE" \
        CACHEGEN_MODEL_NAME="$CACHEGEN_MODEL_NAME" \
        CACHEGEN_INCLUDE_ENCODE_TIME="$CACHEGEN_INCLUDE_ENCODE_TIME" \
        CACHEGEN_DECODE_MS="$CACHEGEN_DECODE_MS" \
        CACHEGEN_SEGMENT_COUNT_MODE="$CACHEGEN_SEGMENT_COUNT_MODE" \
        bash "$SCRIPT_DIR/run_qmsum_cachegen_compare.sh"
    ) &

    pids+=("$!")
    labels+=("$label")
done

overall_status=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    label="${labels[$i]}"
    if wait "$pid"; then
        statuses+=("OK")
    else
        statuses+=("FAIL")
        overall_status=1
    fi
    echo "$label -> ${statuses[$i]}"
done

{
    echo "============================================================"
    echo " QMSum CacheGen Compare Split Summary"
    echo "============================================================"
    for i in "${!labels[@]}"; do
        echo "${labels[$i]}: ${statuses[$i]}"
    done
    echo ""
    echo "Log root: $LOG_ROOT"
    echo "Output tags:"
    for i in "${!labels[@]}"; do
        label="${labels[$i]}"
        docs_part="${label%_gpu*}"
        echo "  cachegen_compare_${docs_part#docs_}_q${MAX_QUERIES}"
    done
} | tee "$SUMMARY_TXT"

exit "$overall_status"
