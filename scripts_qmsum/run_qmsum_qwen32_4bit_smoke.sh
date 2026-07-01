#!/bin/bash
# ============================================================================
# Qwen2.5-32B-Instruct single-GPU 4-bit QMSum smoke.
#
# Use this only after smoke_qwen32_generate.py prints a sane answer. It keeps the
# old Mistral/FastChat path untouched and explicitly enables native HF loading.
# ============================================================================

set -u

MODEL_ID=${MODEL_ID:-~/models/Qwen2.5-32B-Instruct}
DATASET=${DATASET:-qmsum}
if [ -z "${DATA_PATH+x}" ]; then
    if [ "$DATASET" = "hotpotqa" ]; then
        DATA_PATH=~/working_place/topic_selective_kv/data/hotpotqa/validation.jsonl
    else
        DATA_PATH=~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl
    fi
fi
GPU_ID=${GPU_ID:-3}
NGPUS=${NGPUS:-1}
MEM=${MEM:-44}

START_DOC=${START_DOC:-0}
END_DOC=${END_DOC:-1}
MAX_QUERIES=${MAX_QUERIES:-1}
NUM_NODES=${NUM_NODES:-4}
MAX_TOKENS=${MAX_TOKENS:-0}

MAINLINE_PROFILE=${MAINLINE_PROFILE:-current}
TTFT_MODEL=${TTFT_MODEL:-active_node_v2}
CASE_SUMMARY_TAG=${CASE_SUMMARY_TAG:-qwen32_4bit_smoke_${START_DOC}_${END_DOC}_q${MAX_QUERIES}}
LOG_DIR=${LOG_DIR:-logs/qmsum_qwen32_4bit_smoke_${START_DOC}_${END_DOC}_q${MAX_QUERIES}}
LOG_FILE=${LOG_FILE:-"$LOG_DIR/qwen32_4bit.log"}

ANSWER_MAX_NEW_TOKENS=${ANSWER_MAX_NEW_TOKENS:-96}
ANSWER_PROMPT_STYLE=${ANSWER_PROMPT_STYLE:-strict}
SELECTED_ANSWER_CONTEXT_MODE=${SELECTED_ANSWER_CONTEXT_MODE:-turns}
QUERY_TOKENIZER_WARMUP=${QUERY_TOKENIZER_WARMUP:-0}
EVAL_ANSWERS=${EVAL_ANSWERS:-0}

mkdir -p "$LOG_DIR"

extra_args=()
if [ "$MAX_TOKENS" -gt 0 ]; then
    extra_args+=(--max_tokens "$MAX_TOKENS")
fi
if [ "$EVAL_ANSWERS" -eq 1 ]; then
    extra_args+=(
        --eval_answers
        --no_eval_oracle_answers
        --answer_max_new_tokens "$ANSWER_MAX_NEW_TOKENS"
        --answer_prompt_style "$ANSWER_PROMPT_STYLE"
        --selected_answer_context_mode "$SELECTED_ANSWER_CONTEXT_MODE"
    )
fi

echo "============================================================"
echo " QMSum Qwen32B 4-bit smoke"
echo " gpu=$GPU_ID"
echo " dataset=$DATASET"
echo " data=$DATA_PATH"
echo " docs=$START_DOC:$END_DOC"
echo " max_queries=$MAX_QUERIES"
echo " eval_answers=$EVAL_ANSWERS"
echo " profile=$MAINLINE_PROFILE"
echo " ttft_model=$TTFT_MODEL"
echo " model=$MODEL_ID"
echo " log=$LOG_FILE"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU_ID" python qmsum_mainline.py \
    --mainline_profile "$MAINLINE_PROFILE" \
    --ttft_model "$TTFT_MODEL" \
    --dataset "$DATASET" \
    --data_path "$DATA_PATH" \
    --model_id "$MODEL_ID" \
    --model_loader hf \
    --hf_quantization 4bit \
    --hf_dtype bf16 \
    --hf_attn_impl eager \
    --hf_device_map auto \
    --num_gpus "$NGPUS" \
    --max_gpu_memory "$MEM" \
    --num_nodes "$NUM_NODES" \
    --start_doc "$START_DOC" \
    --end_doc "$END_DOC" \
    --max_queries_per_doc "$MAX_QUERIES" \
    --case_summary_tag "$CASE_SUMMARY_TAG" \
    --query_tokenizer_warmup "$QUERY_TOKENIZER_WARMUP" \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG_FILE"

exit "${PIPESTATUS[0]}"
