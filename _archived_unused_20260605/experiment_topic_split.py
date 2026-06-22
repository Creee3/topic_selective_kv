"""
================================================================================
 experiment_topic_split.py — 主实验：按 topic 选择性传输 KV

 做什么：
   把导师的思想跑通一个完整流程：
     1. 完整 prefill → 得到全量 KV（8903 tokens）
     2. topic_boundary 按话题切分 KV
     3. topic_relevance 判断 query 需要哪个 topic
     4. 只保留「系统提示 + 目标 topic + 结尾提问」的 KV
     5. CacheGen 压缩 → 对比全量和筛选后的大小
     6. 推理 → 对比准确率

 两种模式：
   --mode measure: 只测大小节省（不需要 GPU，如果已有 prefill 产物则不需要 model）
   --mode inference: 完整推理对比准确率（需要 GPU + model）

 运行命令：
   # 仅测量大小节省（纯文本，无需 GPU）
   python experiment_topic_split.py --mode measure --doc_id 0

   # 完整推理（需要 GPU + conda cachegen 环境 + 已 prefill 的 KV）
   python experiment_topic_split.py \
       --mode inference \
       --model_id mistralai/Mistral-7B-Instruct-v0.2 \
       --doc_id 0 \
       --num_gpus 1

 预期结果（以 doc_id=0, topic=0 为例）：
   全量 KV:         8903 tokens → ~172 MB (CacheGen 压缩后)
   筛选后 KV:       ~770 tokens → ~15 MB  (仅系统提示+topic0+结尾)
   节省:            ~91%
================================================================================
"""

import sys
import os
import argparse
import pickle
import json
import time

# 把当前目录加入 path，确保能 import 同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topic_boundary import (
    load_longchat_sample,
    find_topic_token_ranges,
    get_system_prompt_range,
    get_final_question_range,
    add_topic_ordinal_markers,
)
from topic_relevance import TopicRelevanceScorer


# ================================================================
# 模式 A: 仅测量大小节省（纯文本分析，不需要 GPU）
# ================================================================
def measure_savings(args):
    """
    不加载模型，不跑推理。只做文本层面的分析：
      - 找到 target topic 的 token 范围
      - 计算全量 vs 筛选后的 token 数和估算压缩大小
    """

    print("=" * 70)
    print("模式 A: 测量大小节省（纯文本分析）")
    print("=" * 70)

    from transformers import AutoTokenizer

    # 加载 tokenizer（CPU 即可）
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    data = load_longchat_sample(args.data_path, doc_id=args.doc_id)

    prompt = data['prompt']
    labels = data['label']

    # 如果开启 markers，先修改 prompt
    if args.use_markers:
        prompt = add_topic_ordinal_markers(prompt)

    query = prompt.split("ASSISTANT:")[-1].strip()  # 提取结尾提问
    # 更好的方式：直接取最后一个 USER 之后的内容
    query = prompt[prompt.rfind("USER:"):].split("ASSISTANT:")[0].strip()

    print(f"\nDoc #{args.doc_id}")
    if args.use_markers:
        print("[Marker 已开启]")
    print(f"Query: \"{query[:80]}...\"")
    print()

    # 1. 检测话题边界
    topics = find_topic_token_ranges(prompt, tokenizer)
    sys_start, sys_end = get_system_prompt_range(prompt, tokenizer)
    q_start, q_end = get_final_question_range(prompt, tokenizer)

    total_tokens = len(tokenizer(prompt).input_ids)
    sys_tokens = sys_end - sys_start
    q_tokens = q_end - q_start

    print(f"总 token 数: {total_tokens}")
    print(f"系统提示:    {sys_tokens} tokens")
    print(f"结尾提问:    {q_tokens} tokens")
    print()

    # 2. 相关性打分
    topic_names = [t['name'] for t in topics]
    scorer = TopicRelevanceScorer(method=args.relevance_method)
    full_query = prompt.split("Now the record ends.")[-1].strip() if "Now the record ends." in prompt else query

    if args.relevance_method == "attention":
        # attention 方法需要模型，measure 模式不支持
        print("⚠ measure 模式不支持 attention 方法，回退到 keyword_match")
        scorer = TopicRelevanceScorer(method="keyword_match")

    best_idx, best_name, best_score, info = scorer.get_best_topic(full_query, topic_names)

    if info.get("matched_topic_idx") is None:
        print("⚠ keyword 未匹配到序数词，使用默认 topic 0")
        best_idx = 0
        best_name = topics[0]['name']

    print(f"★ 相关性判断: {info.get('method', 'unknown')}")
    print(f"  匹配词: '{info.get('matched_word', 'N/A')}' → Topic {best_idx}")
    print(f"  Topic 名称: \"{best_name}\"")
    print()

    # 3. 计算 token 数：全量 vs 筛选后
    selected_tokens = sys_tokens + topics[best_idx]['n_tokens'] + q_tokens

    print("=" * 70)
    print("Token 数对比")
    print("=" * 70)
    print(f"  全量 KV:        {total_tokens:>6} tokens")
    print(f"  系统提示:       {sys_tokens:>6} tokens")
    print(f"  Topic {best_idx} KV:  {topics[best_idx]['n_tokens']:>6} tokens")
    print(f"  结尾提问:       {q_tokens:>6} tokens")
    print(f"  ────────────────────────")
    print(f"  筛选后 KV:      {selected_tokens:>6} tokens")
    print(f"  节省:           {total_tokens - selected_tokens:>6} tokens ({100*(1-selected_tokens/total_tokens):.1f}%)")

    # 4. 估算压缩后大小（基于 CacheGen 的压缩比）
    #    实测：8903 tokens → ~172 MB → 约 19 KB/token
    #    这只是一个粗略估计
    cg_bytes_per_token = 19e3  # ~19 KB per token (实测 ≈ 172MB/8903)
    full_estimated_mb = total_tokens * cg_bytes_per_token / 1e6
    selected_estimated_mb = selected_tokens * cg_bytes_per_token / 1e6

    print()
    print("=" * 70)
    print("估算 CacheGen 压缩后大小")
    print("=" * 70)
    print(f"  全量 KV:       {full_estimated_mb:.1f} MB")
    print(f"  筛选后 KV:     {selected_estimated_mb:.1f} MB")
    print(f"  节省:          {full_estimated_mb - selected_estimated_mb:.1f} MB ({100*(1-selected_tokens/total_tokens):.1f}%)")

    # 5. 列出所有 15 个 topic 的 token 数（供参考）
    print()
    print("=" * 70)
    print("15 个 Topic 的 token 分布")
    print("=" * 70)
    print(f"  {'#':<4} {'Topic Name':<55} {'tokens':>6} {'%':>6}")
    print("  " + "-" * 72)
    for i, t in enumerate(topics):
        pct = t['n_tokens'] / total_tokens * 100
        marker = " ★" if i == best_idx else ""
        print(f"  {i:<4} {t['name']:<55} {t['n_tokens']:>6} {pct:>5.1f}%{marker}")

    # 6. 保存结果到 outputs/
    result = {
        "doc_id": args.doc_id,
        "total_tokens": total_tokens,
        "system_tokens": sys_tokens,
        "query_tokens": q_tokens,
        "selected_topic_idx": best_idx,
        "selected_topic_name": best_name,
        "selected_topic_tokens": topics[best_idx]['n_tokens'],
        "selected_total_tokens": selected_tokens,
        "savings_pct": 100 * (1 - selected_tokens / total_tokens),
        "all_topics": [
            {"idx": i, "name": t['name'], "tokens": t['n_tokens'],
             "token_start": t['token_start'], "token_end": t['token_end']}
            for i, t in enumerate(topics)
        ],
    }

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/analysis_doc{args.doc_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"详细结果已保存到 outputs/analysis_doc{args.doc_id}.json")

    return result


# ================================================================
# 模式 B: 完整推理对比（需要 GPU + model）
# ================================================================
def run_inference(args):
    """
    完整实验：
      1. 加载模型
      2. Prefill 完整 prompt → 全量 KV
      3. split_kv 按 topic 切分
      4. 筛选 KV（系统提示 + target topic + 结尾提问）
      5. CacheGen 压缩 全量 vs 筛选后
      6. 推理 + 算准确率
    """

    print("=" * 70)
    print("模式 B: 完整推理对比")
    print("=" * 70)

    import torch

    # 导入 CacheGen 和 utils（需要 GPU 环境）
    # 注意：服务器上 lmcache 是 pip install -e 安装的，import 路径是 lmcache.xxx
    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
    from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
    from src.utils import (
        define_model_and_tokenizer, to_blob, split_kv, merge_kv,
        calculate_acc, load_testcases, DATASET_TO_PATH, tensor_to_tuple,
    )
    from transformers import AutoTokenizer

    # CacheGen 内部检查 model_name 来选量化配置，只认 HuggingFace 模型 ID
    # 不支持本地路径。所以用 args.cachegen_model_name 传标准名称
    cg_model_name = args.cachegen_model_name

    # 加载 tokenizer（先用 CPU 做边界检测）
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    data = load_longchat_sample(args.data_path, doc_id=args.doc_id)
    prompt = data['prompt']
    labels = data['label']

    # 如果开启 markers，先修改 prompt
    if args.use_markers:
        prompt = add_topic_ordinal_markers(prompt)

    full_query = prompt.split("Now the record ends.")[-1].strip() if "Now the record ends." in prompt else ""

    # 1. 检测边界（纯文本，不需要 GPU）
    topics = find_topic_token_ranges(prompt, tokenizer)
    sys_start, sys_end = get_system_prompt_range(prompt, tokenizer)
    q_start, q_end = get_final_question_range(prompt, tokenizer)
    topic_names_raw = [t['name'] for t in topics]

    # 2. 相关性判断 —— 分两条路
    use_attention = (args.relevance_method == "attention")

    if use_attention:
        # ================================================================
        # Attention 路径: 先加载模型 → prefill → 提取各 topic 的 K → Q·K^T 打分
        # ================================================================
        print("加载模型 (attention 打分需要模型)...")
        model, tokenizer2 = define_model_and_tokenizer(
            args.model_id,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
        )
        print("模型加载完毕\n")

        # Prefill 完整 prompt
        print("Prefill 完整 prompt...")
        input_ids = tokenizer2(prompt, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=1, return_dict_in_generate=True)
        torch.cuda.synchronize()

        kv = generated['past_key_values']
        # 切掉多生成的 1 个 token，squeeze batch 维 → 3D
        kv = list(kv)
        for i in range(len(kv)):
            kv[i] = list(kv[i])
            kv[i][0] = kv[i][0][:, :, :-1][0]
            kv[i][1] = kv[i][1][:, :, :-1][0]
            kv[i] = tuple(kv[i])
        kv = tuple(kv)

        # 提取 15 个 topic 的 K 张量（每层 K 堆叠）
        print("提取各 topic 的 Key 张量...")
        topic_keys_list = []
        for t in topics:
            kv_t = split_kv(kv, t['token_start'], t['token_end'])
            # kv_t: tuple of (K, V) per layer, K shape (1, heads, topic_len, dim)
            k_stacked = torch.stack([layer[0].squeeze(0) for layer in kv_t], dim=0)
            # → (num_layers, num_heads, topic_len, head_dim)
            topic_keys_list.append(k_stacked)

        # Tokenize query → Q · K^T 打分
        query_ids = tokenizer2(full_query, return_tensors="pt").input_ids.cuda()
        from topic_relevance import attention_score
        scores, info = attention_score(query_ids, topic_keys_list, model)
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_name = topic_names_raw[best_idx]

        print(f"★ Attention 打分结果:")
        for i, s in enumerate(scores):
            marker = " ★" if i == best_idx else ""
            print(f"    [{i}] {topic_names_raw[i]:<50} score={s:.4f}{marker}")
        print(f"  Token 范围: [{topics[best_idx]['token_start']}, {topics[best_idx]['token_end']})")
        print()

    else:
        # ================================================================
        # Keyword / Embedding 路径: 不需要模型，先判定 topic 再加载模型
        # ================================================================
        scorer = TopicRelevanceScorer(method=args.relevance_method)
        best_idx, best_name, _, info = scorer.get_best_topic(full_query, topic_names_raw)

        if info.get("matched_topic_idx") is None and args.relevance_method == "keyword_match":
            print("⚠ keyword 未匹配，使用默认 topic 0")
            best_idx = 0
            best_name = topic_names_raw[0]

        print(f"★ Target topic ({args.relevance_method}): [{best_idx}] \"{best_name}\"")
        print(f"  Token 范围: [{topics[best_idx]['token_start']}, {topics[best_idx]['token_end']})")
        print()

        # 加载模型
        print("加载模型...")
        model, tokenizer2 = define_model_and_tokenizer(
            args.model_id,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
        )
        print("模型加载完毕\n")

        # Prefill 完整 prompt
        print("Prefill 完整 prompt...")
        input_ids = tokenizer2(prompt, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=1, return_dict_in_generate=True)
        torch.cuda.synchronize()

        kv = generated['past_key_values']
        # 切掉多生成的 1 个 token，squeeze batch 维 → 3D
        kv = list(kv)
        for i in range(len(kv)):
            kv[i] = list(kv[i])
            kv[i][0] = kv[i][0][:, :, :-1][0]
            kv[i][1] = kv[i][1][:, :, :-1][0]
            kv[i] = tuple(kv[i])
        kv = tuple(kv)

    # --- 下面两路汇合，kv / best_idx / input_ids 已就绪 ---

    # 用模型的 tokenizer 重新检测边界，保证和 input_ids / kv 完全对齐
    # CPU tokenizer 和模型 tokenizer 的 token 数可能不同，导致切出来的范围错位
    topics = find_topic_token_ranges(prompt, tokenizer2)
    sys_start, sys_end = get_system_prompt_range(prompt, tokenizer2)
    q_start, q_end = get_final_question_range(prompt, tokenizer2)
    q_end = min(q_end, input_ids.shape[1])

    kv_tensor = to_blob(kv)  # (32, 2, 8, N, 128) for GQA
    full_tokens = kv_tensor.shape[-2]
    print(f"全量 KV: {full_tokens} tokens")
    print(f"  [验证] full_kv_tokens={full_tokens}, input_ids_tokens={input_ids.shape[1]}, 匹配={full_tokens == input_ids.shape[1]}")
    print()

    # 5. 按 topic 切分 KV
    print("切分 KV...")
    kv_system = split_kv(kv, sys_start, sys_end)
    kv_topic = split_kv(kv, topics[best_idx]['token_start'], topics[best_idx]['token_end'])
    kv_query = split_kv(kv, q_start, q_end)

    # 合并: system + topic + query
    kv_selected = merge_kv(kv_system, kv_topic)
    kv_selected = merge_kv(kv_selected, kv_query)

    # 提前构建全量 KV 的 4D 版本（推理用），然后释放 3D kv
    kv_full_4d = tuple(
        (layer[0].unsqueeze(0), layer[1].unsqueeze(0)) for layer in kv
    )
    del kv_system, kv_topic, kv_query
    del kv  # 释放 tuple 版本，省显存
    torch.cuda.empty_cache()

    kv_selected_squeezed = tuple(
        (k.squeeze(0), v.squeeze(0)) for k, v in kv_selected
    )
    kv_selected_tensor = to_blob(kv_selected_squeezed)
    del kv_selected_squeezed
    selected_tokens = kv_selected_tensor.shape[-2]
    print(f"筛选后 KV: {selected_tokens} tokens")
    print(f"节省: {100*(1-selected_tokens/full_tokens):.1f}%\n")

    # 6. CacheGen 压缩——全量 vs 筛选后
    print("=" * 70)
    print("CacheGen 压缩对比")
    print("=" * 70)

    lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=args.chunk_size)
    meta_data = LMCacheEngineMetadata(
        model_name=cg_model_name, fmt="huggingface", world_size=1, worker_id=0
    )

    # --- 全量压缩 ---
    print("压缩全量 KV...")
    os.environ["QUANT_LEVEL"] = str(args.quant_level)
    full_serializer = CacheGenSerializer(lmcache_config, meta_data)
    t0 = time.perf_counter()
    full_bytes = full_serializer.to_bytes(kv_tensor)
    torch.cuda.synchronize()
    full_encode_time = time.perf_counter() - t0
    full_size_mb = len(full_bytes) / 1e6

    print(f"  全量: {full_size_mb:.1f} MB, 编码耗时 {full_encode_time*1000:.0f} ms")

    # 释放全量 tensor
    del kv_tensor
    torch.cuda.empty_cache()

    # --- 筛选后压缩 ---
    print("压缩筛选后 KV...")
    selected_serializer = CacheGenSerializer(lmcache_config, meta_data)
    t0 = time.perf_counter()
    selected_bytes = selected_serializer.to_bytes(kv_selected_tensor)
    torch.cuda.synchronize()
    selected_encode_time = time.perf_counter() - t0
    selected_size_mb = len(selected_bytes) / 1e6

    print(f"  筛选后: {selected_size_mb:.1f} MB, 编码耗时 {selected_encode_time*1000:.0f} ms")
    print(f"  大小节省: {100*(1-selected_size_mb/full_size_mb):.1f}%")
    print()

    # 7. 推理对比
    print("=" * 70)
    print("推理对比")
    print("=" * 70)

    # --- 全量 KV 推理 ---
    print("全量 KV 推理...")
    full_output = model.generate(
        input_ids,
        past_key_values=kv_full_4d,
        max_new_tokens=20,
    )
    full_pred = tokenizer2.decode(
        full_output[0][input_ids.shape[1]:], skip_special_tokens=True
    )

    # --- 筛选后 KV 推理 ---
    # kv_selected 来自 merge_kv(split_kv(...))，split_kv 内部 unsqueeze(0) 已加 batch 维
    # 直接从 full input_ids 按 token 范围切片，保证和 kv_selected 完全对齐
    sys_ids = input_ids[0, sys_start:sys_end]
    topic_ids = input_ids[0, topics[best_idx]['token_start']:topics[best_idx]['token_end']]
    query_ids = input_ids[0, q_start:q_end]
    selected_input_ids = torch.cat([sys_ids, topic_ids, query_ids], dim=0).unsqueeze(0)

    kv_selected_tokens = kv_selected[0][0].shape[2]
    if selected_input_ids.shape[1] != kv_selected_tokens:
        print(f"  ⚠ 边界对齐仍有偏差: input_ids={selected_input_ids.shape[1]}, kv={kv_selected_tokens}")
        selected_input_ids = selected_input_ids[:, :kv_selected_tokens]

    print("筛选后 KV 推理...")
    # 手动前向传播：model.generate 在 past_key_values 覆盖全部 input_ids 时
    # 可能产生空的 cache_position，导致 IndexError。改用手动逐 token 生成。
    past_kv = kv_selected
    generated_ids = selected_input_ids
    with torch.no_grad():
        for _ in range(20):
            outputs = model.model(
                input_ids=generated_ids[:, -1:],
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = model.lm_head(outputs[0])
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            past_kv = outputs.past_key_values
    selected_pred = tokenizer2.decode(
        generated_ids[0][selected_input_ids.shape[1]:], skip_special_tokens=True
    )

    print(f"\n  全量 KV 预测:  \"{full_pred}\"")
    print(f"  筛选 KV 预测:  \"{selected_pred}\"")

    # 8. 准确率
    if args.dataset_name == "longchat":
        full_acc = calculate_acc(args.dataset_name, full_pred, labels)
        selected_acc = calculate_acc(args.dataset_name, selected_pred, labels)
    else:
        full_acc = calculate_acc(args.dataset_name, full_pred, data)
        selected_acc = calculate_acc(args.dataset_name, selected_pred, data)

    print(f"\n  全量 KV 准确率:  {full_acc}")
    print(f"  筛选 KV 准确率:  {selected_acc}")

    # 9. 保存结果
    result = {
        "doc_id": args.doc_id,
        "relevance_method": args.relevance_method,
        "use_markers": args.use_markers,
        "query": full_query,
        "target_topic": best_idx,
        "target_topic_name": best_name,
        "full_tokens": full_tokens,
        "selected_tokens": selected_tokens,
        "token_savings_pct": 100 * (1 - selected_tokens / full_tokens),
        "full_size_mb": full_size_mb,
        "selected_size_mb": selected_size_mb,
        "size_savings_pct": 100 * (1 - selected_size_mb / full_size_mb),
        "full_encode_ms": full_encode_time * 1000,
        "selected_encode_ms": selected_encode_time * 1000,
        "full_prediction": full_pred,
        "selected_prediction": selected_pred,
        "full_accuracy": full_acc,
        "selected_accuracy": selected_acc,
    }

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/experiment_doc{args.doc_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n结果已保存到 outputs/experiment_doc{args.doc_id}.json")

    return result


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="按 topic 选择性传输 KV Cache 实验"
    )
    p.add_argument("--mode", type=str, default="measure",
                   choices=["measure", "inference"],
                   help="measure=仅分析大小; inference=完整推理对比")
    p.add_argument("--doc_id", type=int, default=0,
                   help="测试第几条 LongChat 样本")
    p.add_argument("--data_path", type=str,
                   default="data/longchat.jsonl")
    p.add_argument("--model_id", type=str,
                   default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_gpu_memory", type=int, default=48)
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--quant_level", type=int, default=2)
    p.add_argument("--cachegen_model_name", type=str,
                   default="mistral-community/Mistral-7B-v0.2",
                   help="CacheGen 内部用的 HuggingFace 模型 ID（和 --model_id 不同时用）")
    p.add_argument("--dataset_name", type=str, default="longchat")
    p.add_argument("--save_dir", type=str, default="outputs")
    p.add_argument("--use_markers", action="store_true", default=False,
                   help="给每个 topic 插入 'This is the Nth topic we discuss.' 标记")
    p.add_argument("--relevance_method", type=str, default="keyword_match",
                   choices=["keyword_match", "embedding", "attention"],
                   help="相关性打分方法")
    args = p.parse_args()

    if args.mode == "measure":
        measure_savings(args)
    elif args.mode == "inference":
        # inference 模式需要 src/utils.py 等（从 CacheGen-main 复制来的）
        # 确保 sys.path 包含 src/
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        run_inference(args)
