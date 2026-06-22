#!/bin/bash
# ============================================================================
# Run the current QMSum mainline as a sharded 4-GPU closeout verification.
#
# Default split:
#   GPU 0 -> docs 0:10,  max_queries=5
#   GPU 1 -> docs 10:20, max_queries=5
#   GPU 2 -> docs 20:30, max_queries=5
#   GPU 3 -> docs 30:40, max_queries=5
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_current_mainline_4gpu_split.sh
#
# Optional overrides:
#   START_DOC=0 END_DOC=20 DOCS_PER_JOB=10 GPUS="0 1" \
#   bash scripts_qmsum/run_qmsum_current_mainline_4gpu_split.sh
# ============================================================================

set -u

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [ "$SCRIPT_DIR" = "$SCRIPT_PATH" ]; then
    SCRIPT_DIR="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"

START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-40}
DOCS_PER_JOB=${DOCS_PER_JOB:-10}
MAX_QUERIES=${MAX_QUERIES:-5}
GPUS=${GPUS:-"0 1 2 3"}
NUM_NODES=${NUM_NODES:-4}
EVAL_ANSWERS=${EVAL_ANSWERS:-1}
RESUME_IF_LOG_OK=${RESUME_IF_LOG_OK:-1}
MAINLINE_PROFILE=${MAINLINE_PROFILE:-current}

LOG_ROOT=${LOG_ROOT:-logs/qmsum_${MAINLINE_PROFILE}_split_${START_DOC}_${END_DOC}_q${MAX_QUERIES}}
SUMMARY_TXT="$LOG_ROOT/summary.txt"

mkdir -p "$LOG_ROOT"

read -r -a GPU_LIST <<< "$GPUS"
GPU_COUNT=${#GPU_LIST[@]}

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPUs configured. Set GPUS=\"0 1 2 3\" or similar."
    exit 1
fi

if [ "$DOCS_PER_JOB" -le 0 ]; then
    echo "DOCS_PER_JOB must be > 0"
    exit 1
fi

pids=()
labels=()
statuses=()

echo "============================================================"
echo " QMSum Current Mainline Split Run"
echo " docs=$START_DOC:$END_DOC"
echo " docs_per_job=$DOCS_PER_JOB"
echo " max_queries=$MAX_QUERIES"
echo " profile=$MAINLINE_PROFILE"
echo " gpus=$GPUS"
echo " num_nodes=$NUM_NODES"
echo " eval_answers=$EVAL_ANSWERS"
echo " log_root=$LOG_ROOT"
echo "============================================================"

job_idx=0
doc_start="$START_DOC"
while [ "$doc_start" -lt "$END_DOC" ]; do
    doc_end=$((doc_start + DOCS_PER_JOB))
    if [ "$doc_end" -gt "$END_DOC" ]; then
        doc_end="$END_DOC"
    fi

    gpu="${GPU_LIST[$((job_idx % GPU_COUNT))]}"
    label="docs_${doc_start}_${doc_end}_gpu${gpu}"
    tag="${MAINLINE_PROFILE}_${doc_start}_${doc_end}_q${MAX_QUERIES}"
    log_dir="$LOG_ROOT/$label"

    echo "Launch $label -> CASE_SUMMARY_TAG=$tag"
    (
        GPU_ID="$gpu" \
        START_DOC="$doc_start" \
        END_DOC="$doc_end" \
        MAX_QUERIES="$MAX_QUERIES" \
        NUM_NODES="$NUM_NODES" \
        EVAL_ANSWERS="$EVAL_ANSWERS" \
        MAINLINE_PROFILE="$MAINLINE_PROFILE" \
        CASE_SUMMARY_TAG="$tag" \
        LOG_DIR="$log_dir" \
        RESUME_IF_LOG_OK="$RESUME_IF_LOG_OK" \
        bash "$SCRIPT_DIR/run_qmsum_current_mainline.sh"
    ) &

    pids+=("$!")
    labels+=("$label")
    doc_start="$doc_end"
    job_idx=$((job_idx + 1))
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
    echo " QMSum Current Mainline Split Summary"
    echo "============================================================"
    for i in "${!labels[@]}"; do
        echo "${labels[$i]}: ${statuses[$i]}"
    done
    echo ""
    echo "Log root: $LOG_ROOT"
    echo "Output tags:"
    doc_start="$START_DOC"
    while [ "$doc_start" -lt "$END_DOC" ]; do
        doc_end=$((doc_start + DOCS_PER_JOB))
        if [ "$doc_end" -gt "$END_DOC" ]; then
            doc_end="$END_DOC"
        fi
        echo "  ${MAINLINE_PROFILE}_${doc_start}_${doc_end}_q${MAX_QUERIES}"
        doc_start="$doc_end"
    done
} | tee "$SUMMARY_TXT"

exit "$overall_status"
