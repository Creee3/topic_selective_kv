"""
================================================================================
 experiment_chunk_split.py — Chunk 级 Q-K Attention 选择性 KV 传输

 核心思路：
   用模型自身的 Q·K^T 注意力信号判断哪些固定大小的 KV chunk 对 query
   最重要。不需要边界检测、不依赖数据集格式。

 流程：
   1. Prefill 完整 prompt → 全量 KV
   2. 按固定 token 数切 chunk（如 256 token/chunk）
   3. 每个 chunk 提 K 张量，和 query 的 Q 算 attention score
   4. 选 top-k chunk + 强制保留首尾 chunk
   5. merge → CacheGen 压缩 → 推理对比

 运行：
   python experiment_chunk_split.py --mode inference \
       --model_id ~/models/mistral-7b/ --doc_id 0 --num_gpus 1 \
       --max_gpu_memory 40 --chunk_size 256 --top_k 3
================================================================================
"""

import sys
import os
import argparse
import json
import time
import re
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 序数词映射（从 topic_relevance.py 复制，避免循环导入）
ORDINAL_TO_NUMBER = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
}
NUMBER_TO_ORDINAL = {v: k for k, v in ORDINAL_TO_NUMBER.items() if not k[0].isdigit()}


# ================================================================
# Chunk Q-K 打分辅助函数
# ================================================================

def _resolve_query_topk(query_len, query_topk_ratio):
    if query_len <= 0:
        return 1
    try:
        ratio = float(query_topk_ratio)
    except (TypeError, ValueError):
        ratio = 0.25
    if ratio <= 0.0:
        ratio = 0.25
    return max(1, min(int(query_len), int(np.ceil(float(query_len) * ratio))))


def _pool_qk_attention(attn, mode="mean", query_topk_ratio=0.25):
    """Pool raw Q-K logits into one score per head."""
    mode = str(mode or "mean").lower()
    if mode == "mean":
        return attn.mean(dim=(2, 3)).squeeze(0)
    if mode == "query_mean_topk":
        per_query = attn.mean(dim=3)
    elif mode == "query_peak_topk":
        per_query = attn.max(dim=3).values
    else:
        raise ValueError(f"Unknown qk token pooling mode: {mode}")

    query_len = int(per_query.shape[-1])
    topk = _resolve_query_topk(query_len, query_topk_ratio)
    return per_query.topk(k=topk, dim=-1).values.mean(dim=-1).squeeze(0)


def _get_qk_attention(
    query_ids,
    chunk_keys_list,
    model,
    layer_idx,
    qk_token_pooling="mean",
    qk_query_topk_ratio=0.25,
):
    """单层 Q-K attention 原始分数，返回 (n_chunks, n_heads) numpy 数组"""
    num_layers = len(model.model.layers)
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    layer = model.model.layers[layer_idx]
    q_proj = layer.self_attn.q_proj
    num_heads = layer.self_attn.num_heads
    head_dim = layer.self_attn.head_dim
    num_kv_heads = getattr(layer.self_attn, 'num_key_value_heads', num_heads)

    with torch.no_grad():
        outputs = model.model(
            input_ids=query_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        # hidden_states[0] = embedding, hidden_states[i] = layer i-1 输出 = layer i 输入
        hidden = outputs.hidden_states[layer_idx]  # layer i 的输入

    Q = q_proj(hidden)
    Q = Q.view(1, -1, num_heads, head_dim).transpose(1, 2)
    # Q: (1, num_heads, query_len, head_dim)

    scale = head_dim ** 0.5
    per_head_scores = []
    for ck in chunk_keys_list:
        # ck: (num_layers, num_kv_heads, chunk_len, head_dim)
        K_layer = ck[layer_idx].unsqueeze(0).to(Q.device)
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            K_layer = K_layer.repeat_interleave(n_rep, dim=1)
        attn = torch.matmul(Q, K_layer.transpose(-2, -1)) / scale
        # attn: (1, num_heads, query_len, chunk_len)
        per_head = _pool_qk_attention(
            attn,
            mode=qk_token_pooling,
            query_topk_ratio=qk_query_topk_ratio,
        )
        per_head_scores.append(per_head.cpu().numpy())

    return np.stack(per_head_scores, axis=0)  # (n_chunks, n_heads)


def build_query_q_cache(query_ids, model, layer_indices):
    """Precompute query Q tensors once for all scoring layers.

    This preserves the scorer's original hidden-state indexing, but avoids
    rerunning the query forward for every candidate batch.
    """
    num_layers = len(model.model.layers)
    normalized_layers = [
        int(lidx) if int(lidx) >= 0 else num_layers + int(lidx)
        for lidx in layer_indices
    ]
    unique_layers = list(dict.fromkeys(normalized_layers))
    cache = {
        "layers": normalized_layers,
        "num_model_layers": int(num_layers),
        "q_by_layer": {},
    }

    with torch.no_grad():
        outputs = model.model(
            input_ids=query_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        for lidx in unique_layers:
            layer = model.model.layers[lidx]
            q_proj = layer.self_attn.q_proj
            num_heads = layer.self_attn.num_heads
            head_dim = layer.self_attn.head_dim
            num_kv_heads = getattr(layer.self_attn, "num_key_value_heads", num_heads)

            # hidden_states[0] = embedding, hidden_states[i] = layer i-1 output.
            hidden = outputs.hidden_states[lidx]
            q_tensor = q_proj(hidden)
            q_tensor = q_tensor.view(1, -1, num_heads, head_dim).transpose(1, 2)
            cache["q_by_layer"][int(lidx)] = {
                "q": q_tensor,
                "num_heads": int(num_heads),
                "num_key_value_heads": int(num_kv_heads),
                "head_dim": int(head_dim),
                "scale": float(head_dim ** 0.5),
            }
    return cache


def _get_qk_attention_from_query_cache(
    query_q_cache,
    chunk_keys_list,
    layer_idx,
    qk_token_pooling="mean",
    qk_query_topk_ratio=0.25,
):
    """Single-layer Q-K scores using precomputed query Q tensors."""
    num_layers = int(query_q_cache.get("num_model_layers", 0) or 0)
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    entry = query_q_cache["q_by_layer"][int(layer_idx)]
    Q = entry["q"]
    num_heads = int(entry["num_heads"])
    num_kv_heads = int(entry["num_key_value_heads"])
    scale = float(entry["scale"])

    per_head_scores = []
    for ck in chunk_keys_list:
        K_layer = ck[int(layer_idx)].unsqueeze(0).to(Q.device)
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            K_layer = K_layer.repeat_interleave(n_rep, dim=1)
        attn = torch.matmul(Q, K_layer.transpose(-2, -1)) / scale
        per_head = _pool_qk_attention(
            attn,
            mode=qk_token_pooling,
            query_topk_ratio=qk_query_topk_ratio,
        )
        per_head_scores.append(per_head.cpu().numpy())

    return np.stack(per_head_scores, axis=0)


def _chunk_qk_scores_per_head(
    query_ids,
    chunk_keys_list,
    model,
    layer_indices,
    qk_token_pooling="mean",
    qk_query_topk_ratio=0.25,
):
    """多层逐头打分，返回 (n_chunks, n_heads) numpy + info"""
    all_scores = []
    for lidx in layer_indices:
        scores = _get_qk_attention(
            query_ids,
            chunk_keys_list,
            model,
            lidx,
            qk_token_pooling=qk_token_pooling,
            qk_query_topk_ratio=qk_query_topk_ratio,
        )
        all_scores.append(scores)

    avg_scores = np.mean(all_scores, axis=0)  # (n_chunks, n_heads)
    return avg_scores, {
        "layers": layer_indices,
        "n_heads": avg_scores.shape[1],
        "qk_token_pooling": str(qk_token_pooling),
        "qk_query_topk_ratio": float(qk_query_topk_ratio),
    }


def _chunk_qk_scores_per_head_cached(
    query_q_cache,
    chunk_keys_list,
    model,
    layer_indices,
    qk_token_pooling="mean",
    qk_query_topk_ratio=0.25,
):
    """Multi-layer per-head scores using precomputed query Q tensors."""
    num_layers = int(query_q_cache.get("num_model_layers", len(model.model.layers)) or 0)
    normalized_layers = [
        int(lidx) if int(lidx) >= 0 else num_layers + int(lidx)
        for lidx in layer_indices
    ]
    all_scores = []
    for lidx in normalized_layers:
        scores = _get_qk_attention_from_query_cache(
            query_q_cache,
            chunk_keys_list,
            lidx,
            qk_token_pooling=qk_token_pooling,
            qk_query_topk_ratio=qk_query_topk_ratio,
        )
        all_scores.append(scores)

    avg_scores = np.mean(all_scores, axis=0)
    return avg_scores, {
        "layers": normalized_layers,
        "n_heads": avg_scores.shape[1],
        "query_q_cache": True,
        "qk_token_pooling": str(qk_token_pooling),
        "qk_query_topk_ratio": float(qk_query_topk_ratio),
    }


def _chunk_qk_scores_mean(query_ids, chunk_keys_list, model, layer_indices):
    """多层平均后返回一维 scores list (兼容原始 attention_score 接口)"""
    scores_2d, info = _chunk_qk_scores_per_head(
        query_ids, chunk_keys_list, model, layer_indices
    )
    mean_scores = scores_2d.mean(axis=1)  # (n_chunks,)
    # 归一化到 [0, 1]
    s = mean_scores - mean_scores.min()
    s = s / (s.max() + 1e-8)
    return s.tolist(), info


def run_inference(args):
    print("=" * 70)
    print("Chunk 级 Q-K Attention 选择性 KV 传输")
    print(f"  chunk_size={args.chunk_size}, top_k={args.top_k}")
    print(f"  keep_first={args.keep_first_chunks}, keep_last={args.keep_last_chunks}")
    print("=" * 70)

    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
    from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
    from src.utils import (
        define_model_and_tokenizer, to_blob, split_kv,
        calculate_acc, DATASET_TO_PATH,
    )
    from topic_boundary import load_longchat_sample

    cg_model_name = args.cachegen_model_name

    # ---- 1. 加载数据 ----
    data = load_longchat_sample(args.data_path, doc_id=args.doc_id)
    prompt = data['prompt']
    labels = data['label']
    # 提取结尾 query（"Now the record ends." 之后的内容）
    full_query = prompt.split("Now the record ends.")[-1].strip() if "Now the record ends." in prompt else ""

    # 覆盖 query 中的序数词（测试不同 topic 的 attention 打分能力）
    if args.target_ordinal:
        target_num = ORDINAL_TO_NUMBER.get(args.target_ordinal.lower())
        if target_num is None:
            try:
                target_num = int(args.target_ordinal)
            except ValueError:
                raise ValueError(f"无法解析序数词: {args.target_ordinal}")

        target_word = NUMBER_TO_ORDINAL.get(target_num, args.target_ordinal)
        original_word = None
        for word in ORDINAL_TO_NUMBER:
            if word in full_query.lower():
                original_word = word
                break
        if original_word:
            full_query = full_query.replace(original_word, target_word)
            prompt = prompt.replace(
                f"the {original_word} topic", f"the {target_word} topic"
            )
            print(f"[Query 覆盖] '{original_word}' → '{target_word}'")
            print(f"  新 query: {full_query[:80]}...")
        else:
            print(f"⚠ 未在原 query 中找到序数词，无法覆盖")

    # ---- 2. 加载模型 ----
    print("加载模型...")
    model, tokenizer2 = define_model_and_tokenizer(
        args.model_id,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
    )
    print("模型加载完毕\n")

    # ---- 3. Prefill 完整 prompt ----
    # 不用 model.generate (会多生成 1 token 导致 KV 和 input_ids 对不齐)
    # 直接用 model.model() 做一次前向传播拿到干净的 KV cache
    print("Prefill 完整 prompt...")
    inputs = tokenizer2(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    torch.cuda.synchronize()

    # outputs.past_key_values: tuple of (K, V) per layer
    #   K/V shape: (batch=1, num_heads, seq_len, head_dim)
    # squeeze batch dim → 3D (num_heads, seq_len, head_dim)
    kv = tuple(
        (layer[0][0], layer[1][0]) for layer in outputs.past_key_values
    )

    total_tokens = kv[0][0].shape[1]
    full_tokens = total_tokens
    print(f"全量 KV: {full_tokens} tokens")
    print(f"  input_ids 匹配: {full_tokens == input_ids.shape[1]}")
    print()

    # ---- 4. 按固定大小切 chunk ----
    chunk_size = args.chunk_size
    num_chunks = total_tokens // chunk_size
    if total_tokens % chunk_size != 0:
        num_chunks += 1  # 最后一个 chunk 包含余数

    print(f"切分为 {num_chunks} 个 chunk (chunk_size={chunk_size})")
    if total_tokens % chunk_size != 0:
        print(f"  最后一个 chunk 大小: {total_tokens % chunk_size} tokens (含余数)")

    # 提取每个 chunk 的 K 张量
    chunk_keys_list = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_tokens)
        kv_chunk = split_kv(kv, start, end)
        # kv_chunk: tuple of (K, V) per layer, K shape (1, heads, chunk_len, dim)
        k_stacked = torch.stack([layer[0].squeeze(0) for layer in kv_chunk], dim=0)
        # → (num_layers, num_kv_heads, chunk_len, head_dim)
        chunk_keys_list.append(k_stacked)

    # ---- 5. 确定必须保留的 chunk ----
    first = args.keep_first_chunks
    last = args.keep_last_chunks

    if first + last >= num_chunks:
        # 全保留
        print(f"chunk 数({num_chunks}) <= 必须保留({first}+{last})，全量传输\n")
        selected_indices = list(range(num_chunks))
    else:
        middle_indices = list(range(first, num_chunks - last))
        n_middle = len(middle_indices)

        print(f"  强制保留: front [{0}..{first-1}] + tail [{num_chunks-last}..{num_chunks-1}]")
        print(f"  中间候选: [{first}..{num_chunks-last-1}] = {n_middle} chunks")

        # ---- 6. Q-K Attention 打分 ----
        print("\n计算 Q-K Attention 分数...")
        query_ids = tokenizer2(full_query, return_tensors="pt").input_ids.cuda()
        middle_keys = [chunk_keys_list[i] for i in middle_indices]

        # 确定用哪些层
        num_model_layers = len(model.model.layers)
        if args.scoring_layers:
            scoring_layers = [int(x) for x in args.scoring_layers.split(",")]
        else:
            # 默认: 浅层 + 中层 + 深层各取几层
            scoring_layers = [0, 8, 16, 24, num_model_layers - 1]
        scoring_layers = [l if l >= 0 else num_model_layers + l for l in scoring_layers]

        from topic_relevance import attention_score

        if args.per_head or args.adaptive_threshold:
            # ---- 逐头打分 (返回 (n_chunks, n_heads) ) ----
            scores_matrix, score_info = _chunk_qk_scores_per_head(
                query_ids, middle_keys, model, scoring_layers
            )
            print(f"  打分维度: {scores_matrix.shape[0]} chunks × {scores_matrix.shape[1]} heads")
            print(f"  使用层: {scoring_layers}")
            n_heads = scores_matrix.shape[1]

            if args.per_head:
                selected_middle = set()
                head_votes = np.zeros(n_middle, dtype=int)

                if args.adaptive_threshold:
                    # 逐头 + 阈值: 每个 head 用比例阈值独立选
                    for h in range(n_heads):
                        head_scores = scores_matrix[:, h]
                        score_range = head_scores.max() - head_scores.min()
                        threshold_h = head_scores.max() - args.alpha * score_range
                        for j, s in enumerate(head_scores):
                            if s >= threshold_h:
                                head_votes[j] += 1
                                selected_middle.add(middle_indices[j])
                else:
                    # 逐头 + 固定 top-k
                    k_per_head = max(1, args.top_k) if args.top_k > 0 else 1
                    for h in range(n_heads):
                        head_scores = scores_matrix[:, h]
                        top_k_local = np.argsort(head_scores)[-k_per_head:][::-1]
                        for j in top_k_local:
                            head_votes[j] += 1
                        selected_middle.update(middle_indices[j] for j in top_k_local)

                selected_middle = sorted(selected_middle)
                print(f"  逐头选择: {len(selected_middle)} chunks 被至少 1 个 head 选中")
                for j, (chunk_idx, votes) in enumerate(zip(middle_indices, head_votes)):
                    if votes > 0:
                        bar = "█" * votes
                        print(f"    [{chunk_idx:<6}] {votes:>2}/{n_heads} {bar}")

            elif args.adaptive_threshold:
                # 平均后阈值: score >= max - alpha * range
                mean_scores = scores_matrix.mean(axis=1)
                score_range = mean_scores.max() - mean_scores.min()
                threshold = mean_scores.max() - args.alpha * score_range
                selected_middle = sorted([
                    middle_indices[j] for j, s in enumerate(mean_scores)
                    if s >= threshold
                ])
                print(f"  比例阈值: max={mean_scores.max():.4f}, range={score_range:.4f}, "
                      f"alpha={args.alpha}, threshold={threshold:.4f} → {len(selected_middle)} chunks")
                print(f"\n  {'Chunk':<8} {'Score':<14} {'Select':<8}")
                print("  " + "-" * 30)
                for j, (chunk_idx, s) in enumerate(zip(middle_indices, mean_scores)):
                    mark = "✓" if s >= threshold else ""
                    print(f"  [{chunk_idx:<6}] {s:<14.6f} {mark}")
        else:
            # ---- 原始: 所有 head 平均，单层或平均多层 ----
            scores, info = attention_score(query_ids, middle_keys, model) if len(scoring_layers) == 1 else _chunk_qk_scores_mean(
                query_ids, middle_keys, model, scoring_layers
            )

            print(f"\n  {'Chunk':<8} {'Score':<14}")
            print("  " + "-" * 22)
            for j, (chunk_idx, score) in enumerate(zip(middle_indices, scores)):
                print(f"  [{chunk_idx:<6}] {score:<14.6f}")

            k = min(args.top_k, n_middle)
            top_k_local = np.argsort(scores)[-k:][::-1]
            selected_middle = sorted([middle_indices[j] for j in top_k_local])

        if not args.per_head and not args.adaptive_threshold:
            print(f"\n  选中 {len(selected_middle)}/{n_middle} 个中间 chunk: {selected_middle}")

        selected_indices = (
            list(range(first)) +
            selected_middle +
            list(range(num_chunks - last, num_chunks))
        )
        selected_indices = sorted(set(selected_indices))

    # ---- 7. 构建 selected_input_ids + 重新 prefill ----
    # 不用 merge_kv 拼非连续 KV chunk（RoPE 位置编码会乱）
    # 改为把 selected_input_ids 重新做一次 prefill，位置天然连续
    selected_parts = []
    for i in selected_indices:
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_tokens)
        selected_parts.append(input_ids[0, start:end])
    selected_input_ids = torch.cat(selected_parts, dim=0).unsqueeze(0)

    print(f"\n合并 {len(selected_indices)}/{num_chunks} 个 chunk → {selected_input_ids.shape[1]} tokens")
    print("重新 prefill 筛选文本...")
    with torch.no_grad():
        sel_outputs = model.model(
            input_ids=selected_input_ids,
            use_cache=True,
        )
    kv_selected = tuple(
        (layer[0][0], layer[1][0]) for layer in sel_outputs.past_key_values
    )

    selected_tokens = kv_selected[0][0].shape[1]
    print(f"筛选后 KV: {selected_tokens} tokens (位置连续)")

    # 全量 KV 4D 版本（推理用）
    kv_full_4d = tuple(
        (layer[0].unsqueeze(0), layer[1].unsqueeze(0)) for layer in kv
    )

    # ---- 8. CacheGen 压缩对比 ----
    kv_tensor = to_blob(kv)
    # kv_selected 来自重新 prefill，已 3D，直接 to_blob
    kv_selected_tensor = to_blob(kv_selected)

    print(f"\n筛选后 KV: {selected_tokens} tokens")
    print(f"节省: {100 * (1 - selected_tokens / full_tokens):.1f}%")

    print("\n" + "=" * 70)
    print("CacheGen 压缩对比")
    print("=" * 70)

    lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=args.chunk_size)
    meta_data = LMCacheEngineMetadata(
        model_name=cg_model_name, fmt="huggingface", world_size=1, worker_id=0
    )
    os.environ["QUANT_LEVEL"] = str(args.quant_level)

    # 全量压缩
    print("压缩全量 KV...")
    full_serializer = CacheGenSerializer(lmcache_config, meta_data)
    t0 = time.perf_counter()
    full_bytes = full_serializer.to_bytes(kv_tensor)
    torch.cuda.synchronize()
    full_time = time.perf_counter() - t0
    full_mb = len(full_bytes) / 1e6
    print(f"  全量:   {full_mb:.1f} MB, {full_time * 1000:.0f} ms")

    del kv_tensor
    torch.cuda.empty_cache()

    # 筛选后压缩
    print("压缩筛选后 KV...")
    selected_serializer = CacheGenSerializer(lmcache_config, meta_data)
    t0 = time.perf_counter()
    selected_bytes = selected_serializer.to_bytes(kv_selected_tensor)
    torch.cuda.synchronize()
    selected_time = time.perf_counter() - t0
    selected_mb = len(selected_bytes) / 1e6
    print(f"  筛选后: {selected_mb:.1f} MB, {selected_time * 1000:.0f} ms")
    print(f"  大小节省: {100 * (1 - selected_mb / full_mb):.1f}%")

    del kv_selected_tensor
    torch.cuda.empty_cache()

    # ---- 9. 推理对比 ----
    print("\n" + "=" * 70)
    print("推理对比")
    print("=" * 70)

    # 全量 KV 推理 — 手动前向传播，避免 generate 的 cache_position 为空
    print("全量 KV 推理...")
    full_past = kv_full_4d
    full_gen = input_ids
    with torch.no_grad():
        for _ in range(20):
            outputs = model.model(
                input_ids=full_gen[:, -1:],
                past_key_values=full_past,
                use_cache=True,
            )
            logits = model.lm_head(outputs[0])
            next_token = logits[:, -1:, :].argmax(dim=-1)
            full_gen = torch.cat([full_gen, next_token], dim=-1)
            full_past = outputs.past_key_values
            if next_token.item() == tokenizer2.eos_token_id:
                break
    full_pred = tokenizer2.decode(
        full_gen[0][input_ids.shape[1]:], skip_special_tokens=True
    )

    # 筛选 KV 推理 — selected_input_ids 和 kv_selected 来自同一段重新 prefill，天然对齐
    kv_sel_4d = tuple(
        (layer[0].unsqueeze(0), layer[1].unsqueeze(0)) for layer in kv_selected
    )
    print("筛选后 KV 推理...")
    past_kv = kv_sel_4d
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
            if next_token.item() == tokenizer2.eos_token_id:
                break
    selected_pred = tokenizer2.decode(
        generated_ids[0][selected_input_ids.shape[1]:], skip_special_tokens=True
    )

    print(f"\n  全量 KV 预测:  \"{full_pred}\"")
    print(f"  筛选 KV 预测:  \"{selected_pred}\"")

    # ---- 10. 准确率 ----
    if args.dataset_name == "longchat":
        if args.target_ordinal:
            target_num = ORDINAL_TO_NUMBER.get(args.target_ordinal.lower())
            if target_num is None:
                target_num = int(args.target_ordinal)
            target_label = [labels[target_num - 1]]
        else:
            target_label = labels
        full_acc = calculate_acc(args.dataset_name, full_pred, target_label)
        selected_acc = calculate_acc(args.dataset_name, selected_pred, target_label)
    else:
        full_acc = calculate_acc(args.dataset_name, full_pred, data)
        selected_acc = calculate_acc(args.dataset_name, selected_pred, data)

    print(f"\n  全量 KV 准确率:  {full_acc}")
    print(f"  筛选 KV 准确率:  {selected_acc}")

    # ---- 11. 保存 ----
    result = {
        "doc_id": args.doc_id,
        "chunk_size": chunk_size,
        "top_k": args.top_k,
        "keep_first": first,
        "keep_last": last,
        "num_chunks": num_chunks,
        "selected_indices": selected_indices,
        "full_tokens": full_tokens,
        "selected_tokens": selected_tokens,
        "token_savings_pct": 100 * (1 - selected_tokens / full_tokens),
        "full_size_mb": full_mb,
        "selected_size_mb": selected_mb,
        "size_savings_pct": 100 * (1 - selected_mb / full_mb),
        "full_prediction": full_pred,
        "selected_prediction": selected_pred,
        "full_accuracy": full_acc,
        "selected_accuracy": selected_acc,
    }

    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/chunk_experiment_doc{args.doc_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n结果已保存到 outputs/chunk_experiment_doc{args.doc_id}.json")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Chunk 级 Q-K Attention 选择性 KV 传输"
    )
    p.add_argument("--mode", type=str, default="inference")
    p.add_argument("--doc_id", type=int, default=0)
    p.add_argument("--data_path", type=str, default="data/longchat.jsonl")
    p.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_gpu_memory", type=int, default=48)
    p.add_argument("--chunk_size", type=int, default=256,
                   help="每个 chunk 的 token 数")
    p.add_argument("--top_k", type=int, default=3,
                   help="从中间 chunk 中选多少个")
    p.add_argument("--keep_first_chunks", type=int, default=1,
                   help="强制保留开头几个 chunk")
    p.add_argument("--keep_last_chunks", type=int, default=1,
                   help="强制保留结尾几个 chunk")
    p.add_argument("--quant_level", type=int, default=2)
    p.add_argument("--cachegen_model_name", type=str,
                   default="mistral-community/Mistral-7B-v0.2")
    p.add_argument("--dataset_name", type=str, default="longchat")
    p.add_argument("--target_ordinal", type=str, default=None,
                   help="覆盖 query 中的序数词，测试不同 topic (如 'eighth', '8th', '5')")
    p.add_argument("--per_head", action="store_true", default=False,
                   help="逐头独立选 chunk，取并集（Quest 风格）")
    p.add_argument("--adaptive_threshold", action="store_true", default=False,
                   help="阈值自适应选择，替代固定 top-k（InfiniGen 风格）")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="自适应阈值比例: threshold = max - alpha * range (默认 0.05)")
    p.add_argument("--scoring_layers", type=str, default=None,
                   help="用哪些层打分，逗号分隔 (如 '0,8,16,24,31')，默认: 0,8,16,24,最后一层")

    args = p.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    run_inference(args)
