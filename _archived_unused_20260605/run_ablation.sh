#!/bin/bash
# ============================================================================
# LongChat 消融实验批量运行脚本
# 用法: bash run_ablation.sh
# 在云端: cd ~/working_place/topic_selective_kv && bash run_ablation.sh
#
# 输出文件命名: batch_longchat_{策略}_k{topk}_c{chunk}_{层数}l.json
#   策略: ph=per_head, th=threshold, mean=平均池化
#   层数: 5l=5层打分, 1l=单层打分
# ============================================================================

MODEL_ID=~/models/mistral-7b/
NGPUS=1
MEM=40
DATASET=longchat
N_SAMPLES=50

echo "============================================"
echo " LongChat 消融实验矩阵 (n=$N_SAMPLES)"
echo " 预计 9 组实验"
echo "============================================"

# ============================================================================
# 1. top_k 扫描 — 准确率 vs 节省的 tradeoff 曲线
#    per_head, chunk=256, 5层打分
# ============================================================================
echo ""
echo "=========================================="
echo " [1/9] top_k=1 — 最激进节省"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 1 --keep_first_chunks 0 --keep_last_chunks 0 \
    --end_doc $N_SAMPLES

echo ""
echo "=========================================="
echo " [2/9] top_k=2 — 均衡点"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 2 --keep_first_chunks 0 --keep_last_chunks 0 \
    --end_doc $N_SAMPLES

echo ""
echo "=========================================="
echo " [3/9] top_k=3 — 偏保守"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 3 --keep_first_chunks 0 --keep_last_chunks 0 \
    --end_doc $N_SAMPLES

echo ""
echo "=========================================="
echo " [4/9] top_k=4 — 最保守"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 4 --keep_first_chunks 0 --keep_last_chunks 0 \
    --end_doc $N_SAMPLES

# ============================================================================
# 2. chunk_size 扫描 — 粒度对准确率的影响
#    per_head, top_k=2, 5层打分
# ============================================================================
echo ""
echo "=========================================="
echo " [5/9] chunk_size=128 — 细粒度"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 2 --chunk_size 128 --keep_first_chunks 0 --keep_last_chunks 0 \
    --end_doc $N_SAMPLES

echo ""
echo "=========================================="
echo " [6/9] chunk_size=512 — 粗粒度"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 2 --chunk_size 512 --keep_first_chunks 0 --keep_last_chunks 0 \
    --end_doc $N_SAMPLES

# ============================================================================
# 3. 打分策略消融 — per_head 贡献了多少？
#    chunk=256, top_k=2, 5层打分
# ============================================================================
echo ""
echo "=========================================="
echo " [7/9] 打分策略: mean pooling — 无 per_head"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --top_k 2 --keep_first_chunks 0 --keep_last_chunks 0 \
    --end_doc $N_SAMPLES

echo ""
echo "=========================================="
echo " [8/9] 打分策略: adaptive_threshold + per_head"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --adaptive_threshold --alpha 0.05 --top_k 2 \
    --keep_first_chunks 0 --keep_last_chunks 0 \
    --end_doc $N_SAMPLES

# ============================================================================
# 4. 打分层的消融 — 多层是否比单层好？
#    per_head, chunk=256, top_k=2, 仅用最后一层
# ============================================================================
echo ""
echo "=========================================="
echo " [9/9] 打分层的消融: 单层 (layer 31 only)"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 2 --scoring_layers "31" \
    --keep_first_chunks 0 --keep_last_chunks 0 \
    --end_doc $N_SAMPLES

echo ""
echo "============================================"
echo " 全部消融实验完成！"
echo ""
echo " 查看结果: python analyze_ablation.py"
echo "============================================"
