#!/bin/bash
# ============================================================================
# LongChat 多位置 topic 检索能力测试
# 测试 selective KV 对非末尾 topic 的检索效果
#
# 用法: bash run_topic_ablation.sh
# 在云端: cd ~/working_place/topic_selective_kv && bash run_topic_ablation.sh
# ============================================================================

MODEL_ID=~/models/mistral-7b/
NGPUS=1
MEM=40
DATASET=longchat
N_SAMPLES=50

echo "============================================"
echo " LongChat 多位置 topic 检索测试 (n=$N_SAMPLES)"
echo " per_head, top_k=2, c256, 5层打分"
echo "============================================"
echo ""
echo "测试目标: 证明 selective KV 对不同位置 topic 的检索能力"
echo "关键指标: 全量 vs 筛选预测一致性 (而非 label 准确率)"
echo ""

# ---- first topic (开头，最容易) ----
echo "=========================================="
echo " [1/3] target_ordinal=first"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 2 --keep_first_chunks 0 --keep_last_chunks 0 \
    --target_ordinal first --end_doc $N_SAMPLES

# ---- third topic (中间偏前) ----
echo ""
echo "=========================================="
echo " [2/3] target_ordinal=third"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 2 --keep_first_chunks 0 --keep_last_chunks 0 \
    --target_ordinal third --end_doc $N_SAMPLES

# ---- fifth topic (中间位置) ----
echo ""
echo "=========================================="
echo " [3/3] target_ordinal=fifth"
echo "=========================================="
python batch_eval.py --dataset $DATASET --model_id $MODEL_ID \
    --num_gpus $NGPUS --max_gpu_memory $MEM \
    --per_head --top_k 2 --keep_first_chunks 0 --keep_last_chunks 0 \
    --target_ordinal fifth --end_doc $N_SAMPLES

echo ""
echo "============================================"
echo " 多位置 topic 测试完成！"
echo ""
echo " 输出文件:"
echo "   outputs/batch_longchat_ph_k2_c256_5l_ordfirst.json"
echo "   outputs/batch_longchat_ph_k2_c256_5l_ordthird.json"
echo "   outputs/batch_longchat_ph_k2_c256_5l_ordfifth.json"
echo ""
echo " 对照基线 (末尾 topic, 已跑过):"
echo "   outputs/batch_longchat_ph_k2_c256_5l.json"
echo "============================================"
