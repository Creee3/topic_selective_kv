#!/bin/bash
# ============================================================================
# Wait for one or more idle GPUs, then run a command with CUDA_VISIBLE_DEVICES.
#
# This script is intentionally non-destructive:
#   - it only reads nvidia-smi status
#   - it never kills processes
#   - it never resets GPUs
#
# Example:
#   bash scripts_qmsum/wait_for_free_gpu.sh \
#     --gpus "0 1 2 3" \
#     --max-mem-mib 2000 \
#     --max-util 10 \
#     -- bash scripts_qmsum/run_qmsum_current_mainline.sh
# ============================================================================

set -euo pipefail

GPU_LIST=${GPU_LIST:-"0 1 2 3"}
NUM_GPUS=${NUM_GPUS:-1}
MAX_MEM_MIB=${MAX_MEM_MIB:-2000}
MAX_UTIL=${MAX_UTIL:-10}
CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC:-60}
TIMEOUT_SEC=${TIMEOUT_SEC:-0}
DRY_RUN=${DRY_RUN:-0}

usage() {
    cat <<'EOF'
Usage:
  wait_for_free_gpu.sh [options] -- command [args...]

Options:
  --gpus "0 1 2 3"       Candidate physical GPU ids. Default: GPU_LIST or "0 1 2 3"
  --num-gpus N           Number of idle GPUs required. Default: NUM_GPUS or 1
  --max-mem-mib M        GPU is idle if memory.used <= M. Default: MAX_MEM_MIB or 2000
  --max-util P           GPU is idle if utilization.gpu <= P. Default: MAX_UTIL or 10
  --interval SEC         Poll interval. Default: CHECK_INTERVAL_SEC or 60
  --timeout SEC          0 means wait forever. Default: TIMEOUT_SEC or 0
  --dry-run              Print selected GPUs and command without running it
  -h, --help             Show this help

Exports before running the command:
  CUDA_VISIBLE_DEVICES   comma-separated selected physical GPU ids
  GPU_ID                 first selected GPU id, for single-GPU scripts
  GPUS                   space-separated selected GPU ids, for split scripts
  SELECTED_GPU_IDS       comma-separated selected physical GPU ids
EOF
}

cmd=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --max-mem-mib)
            MAX_MEM_MIB="$2"
            shift 2
            ;;
        --max-util)
            MAX_UTIL="$2"
            shift 2
            ;;
        --interval)
            CHECK_INTERVAL_SEC="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SEC="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            cmd=("$@")
            break
            ;;
        *)
            cmd=("$@")
            break
            ;;
    esac
done

if [ "${#cmd[@]}" -eq 0 ]; then
    usage
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found." >&2
    exit 127
fi

if [ "$NUM_GPUS" -le 0 ]; then
    echo "ERROR: --num-gpus must be > 0." >&2
    exit 2
fi

start_ts=$(date +%s)

contains_gpu() {
    local needle="$1"
    local item
    for item in $GPU_LIST; do
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

join_by_space() {
    local IFS=' '
    echo "$*"
}

while true; do
    mapfile -t gpu_rows < <(
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
    )

    selected=()
    status_lines=()
    for row in "${gpu_rows[@]}"; do
        row=${row// /}
        IFS=',' read -r gpu_id mem_used util_gpu <<<"$row"
        if ! contains_gpu "$gpu_id"; then
            continue
        fi
        status_lines+=("gpu${gpu_id}:mem=${mem_used}MiB,util=${util_gpu}%")
        if [ "$mem_used" -le "$MAX_MEM_MIB" ] && [ "$util_gpu" -le "$MAX_UTIL" ]; then
            selected+=("$gpu_id")
        fi
    done

    if [ "${#selected[@]}" -ge "$NUM_GPUS" ]; then
        selected=("${selected[@]:0:$NUM_GPUS}")
        selected_csv=$(join_by_comma "${selected[@]}")
        selected_space=$(join_by_space "${selected[@]}")
        echo "Selected GPU(s): $selected_space"
        echo "Command: ${cmd[*]}"

        export CUDA_VISIBLE_DEVICES="$selected_csv"
        export GPU_ID="${selected[0]}"
        export GPUS="$selected_space"
        export SELECTED_GPU_IDS="$selected_csv"

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "DRY_RUN=1, command not executed."
            exit 0
        fi

        exec "${cmd[@]}"
    fi

    now_ts=$(date +%s)
    elapsed=$((now_ts - start_ts))
    echo "No idle GPU yet after ${elapsed}s. Need ${NUM_GPUS}; thresholds: mem<=${MAX_MEM_MIB}MiB util<=${MAX_UTIL}%."
    echo "Candidates: ${status_lines[*]:-none}"

    if [ "$TIMEOUT_SEC" -gt 0 ] && [ "$elapsed" -ge "$TIMEOUT_SEC" ]; then
        echo "ERROR: timed out waiting for idle GPU." >&2
        exit 124
    fi

    sleep "$CHECK_INTERVAL_SEC"
done
