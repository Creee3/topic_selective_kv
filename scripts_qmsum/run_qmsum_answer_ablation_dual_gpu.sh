#!/bin/bash
# ============================================================================
# Run two answer-interface ablations in parallel on two GPUs.
#
# Goal:
#   isolate whether answer degradation comes from:
#   1. strict prompt style
#   2. chunk_turns evidence format
#
# Default assignment:
#   GPU 2 -> turns + strict
#   GPU 3 -> chunk_turns + basic
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_answer_ablation_dual_gpu.sh
#
# Optional overrides:
#   GPU_A=2 GPU_B=3 START_DOC=5 END_DOC=10 MAX_QUERIES=5 ROUTE_TOP_K=12 \
#   bash scripts_qmsum/run_qmsum_answer_ablation_dual_gpu.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-3}
START_DOC=${START_DOC:-5}
END_DOC=${END_DOC:-10}
MAX_QUERIES=${MAX_QUERIES:-5}
ROUTE_TOP_K=${ROUTE_TOP_K:-12}

echo "============================================================"
echo " QMSum Answer Ablation Dual GPU"
echo " GPU_A=$GPU_A -> turns + strict"
echo " GPU_B=$GPU_B -> chunk_turns + basic"
echo " docs=$START_DOC:$END_DOC"
echo " MAX_QUERIES=$MAX_QUERIES"
echo " ROUTE_TOP_K=$ROUTE_TOP_K"
echo "============================================================"

GPU_ID="$GPU_A" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
DYNAMIC_ROUTE_BUDGET=1 \
DYNAMIC_SUMMARY_TOP_K=16 \
DYNAMIC_DETAIL_TOP_K=8 \
DYNAMIC_BALANCED_TOP_K=12 \
ANSWER_EVIDENCE_ORDER=qk_then_time \
SELECTED_ANSWER_CONTEXT_MODE=turns \
ANSWER_PROMPT_STYLE=strict \
ANSWER_EVIDENCE_MAX_ENTRIES=80 \
ANSWER_EVIDENCE_MAX_CHARS=600 \
CASE_SUMMARY_TAG=answer_strict_turns_5docs_q5 \
LOG_DIR=logs/qmsum_answer_strict_turns_5docs_q5 \
bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh" &
PID_A=$!

GPU_ID="$GPU_B" \
START_DOC="$START_DOC" \
END_DOC="$END_DOC" \
MAX_QUERIES="$MAX_QUERIES" \
ROUTE_TOP_K="$ROUTE_TOP_K" \
DYNAMIC_ROUTE_BUDGET=1 \
DYNAMIC_SUMMARY_TOP_K=16 \
DYNAMIC_DETAIL_TOP_K=8 \
DYNAMIC_BALANCED_TOP_K=12 \
ANSWER_EVIDENCE_ORDER=qk_then_time \
SELECTED_ANSWER_CONTEXT_MODE=chunk_turns \
ANSWER_PROMPT_STYLE=basic \
ANSWER_EVIDENCE_MAX_ENTRIES=80 \
ANSWER_EVIDENCE_MAX_CHARS=600 \
CASE_SUMMARY_TAG=answer_basic_chunkturns_5docs_q5 \
LOG_DIR=logs/qmsum_answer_basic_chunkturns_5docs_q5 \
bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh" &
PID_B=$!

wait $PID_A
STATUS_A=$?

wait $PID_B
STATUS_B=$?

echo ""
echo "============================================================"
echo " Dual GPU ablation finished"
echo " turns + strict      status=$STATUS_A"
echo " chunk_turns + basic status=$STATUS_B"
echo " Logs:"
echo "   logs/qmsum_answer_strict_turns_5docs_q5"
echo "   logs/qmsum_answer_basic_chunkturns_5docs_q5"
echo "============================================================"

if [ "$STATUS_A" -ne 0 ] || [ "$STATUS_B" -ne 0 ]; then
    exit 1
fi
