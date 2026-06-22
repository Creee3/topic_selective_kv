#!/bin/bash
# ============================================================================
# Run the current QMSum mainline answer evaluation on two GPUs in parallel.
#
# Default split:
#   GPU 2 -> docs [0, 10)
#   GPU 3 -> docs [10, 20)
#
# Mainline:
#   lexical coarse routing
#   -> top1 topic
#   -> Q-K fine chunk routing
#   -> route_top_k=12
#   -> compare selective KV vs full KV on answer F1
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_mainline_answer_eval_20docs_dual_gpu.sh
#
# Optional overrides:
#   GPU_A=2 GPU_B=3 MAX_QUERIES=0 \
#   bash scripts_qmsum/run_qmsum_mainline_answer_eval_20docs_dual_gpu.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-3}
MAX_QUERIES=${MAX_QUERIES:-0}
ROUTE_TOP_K=${ROUTE_TOP_K:-12}
NUM_NODES=${NUM_NODES:-4}
HIER_TOP_TOPICS=${HIER_TOP_TOPICS:-1}
MAX_TOKENS=${MAX_TOKENS:-0}
START_A=${START_A:-0}
END_A=${END_A:-10}
START_B=${START_B:-10}
END_B=${END_B:-20}

echo "============================================================"
echo " QMSum Mainline Answer Eval Dual GPU"
echo " GPU_A=$GPU_A docs=$START_A:$END_A"
echo " GPU_B=$GPU_B docs=$START_B:$END_B"
echo " NUM_NODES=$NUM_NODES"
echo " MAX_QUERIES=$MAX_QUERIES"
echo " HIER_TOP_TOPICS=$HIER_TOP_TOPICS"
echo " ROUTE_TOP_K=$ROUTE_TOP_K"
echo " MAX_TOKENS=$MAX_TOKENS"
echo "============================================================"

GPU_ID="$GPU_A" \
START_DOC="$START_A" \
END_DOC="$END_A" \
MAX_QUERIES="$MAX_QUERIES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
NUM_NODES="$NUM_NODES" \
HIER_TOP_TOPICS="$HIER_TOP_TOPICS" \
MAX_TOKENS="$MAX_TOKENS" \
LOG_DIR="logs/qmsum_mainline_answer_eval_${START_A}_${END_A}" \
bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh" &
PID_A=$!

GPU_ID="$GPU_B" \
START_DOC="$START_B" \
END_DOC="$END_B" \
MAX_QUERIES="$MAX_QUERIES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
NUM_NODES="$NUM_NODES" \
HIER_TOP_TOPICS="$HIER_TOP_TOPICS" \
MAX_TOKENS="$MAX_TOKENS" \
LOG_DIR="logs/qmsum_mainline_answer_eval_${START_B}_${END_B}" \
bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh" &
PID_B=$!

wait $PID_A
STATUS_A=$?

wait $PID_B
STATUS_B=$?

echo ""
echo "============================================================"
echo " Dual GPU run finished"
echo " GPU_A status=$STATUS_A"
echo " GPU_B status=$STATUS_B"
echo " Logs:"
echo "   logs/qmsum_mainline_answer_eval_${START_A}_${END_A}"
echo "   logs/qmsum_mainline_answer_eval_${START_B}_${END_B}"
echo "============================================================"

if [ "$STATUS_A" -ne 0 ] || [ "$STATUS_B" -ne 0 ]; then
    exit 1
fi
