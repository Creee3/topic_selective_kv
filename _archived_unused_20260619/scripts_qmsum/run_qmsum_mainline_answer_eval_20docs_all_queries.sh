#!/bin/bash
# ============================================================================
# Current mainline answer evaluation on the first 20 docs with all queries.
#
# This is the larger validation pass you asked for:
#   lexical coarse routing
#   -> top1 topic
#   -> Q-K fine chunk routing
#   -> route_top_k=12
#   -> compare selective KV vs full KV on answer F1
#
# Usage:
#   cd ~/working_place/topic_selective_kv
#   bash scripts_qmsum/run_qmsum_mainline_answer_eval_20docs_all_queries.sh
#
# Optional overrides:
#   GPU_ID=2 bash scripts_qmsum/run_qmsum_mainline_answer_eval_20docs_all_queries.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export START_DOC=${START_DOC:-0}
export END_DOC=${END_DOC:-20}
export MAX_QUERIES=${MAX_QUERIES:-0}
export ROUTE_TOP_K=${ROUTE_TOP_K:-12}
export NUM_NODES=${NUM_NODES:-4}
export HIER_TOP_TOPICS=${HIER_TOP_TOPICS:-1}
export MAX_TOKENS=${MAX_TOKENS:-0}
export LOG_DIR=${LOG_DIR:-logs/qmsum_mainline_answer_eval_20docs_all_queries}

bash "$SCRIPT_DIR/run_qmsum_mainline_answer_eval.sh"
