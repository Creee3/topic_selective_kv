"""
================================================================================
 topic_relevance.py — 查询-话题相关性判断

 做什么：
   来了一个用户查询（query），判断它跟 15 个 topic 中哪一个最相关。
   这是导师方案的核心——在传输/压缩之前，先筛选出相关的 KV chunk。

 三种方案（从简单到复杂）：

   方案 A — keyword_match（规则匹配）
     LongChat 的查询是模板化的："What is the first topic we discussed?"
     直接用正则提取 "first" → topic 0, "third" → topic 2, ...
     简单粗暴但在这个数据集上最有效。

   方案 B — embedding_similarity（语义相似度）
     用 embedding 模型把 query 和 15 个 topic name 都编码成向量，
     算余弦相似度，取最高的。

   方案 C — attention_score（Q-K 注意力分数）★ 导师提到的方法
     把 query 文本过一遍模型拿到 Q 张量，
     和每个 topic 的 K 张量算 attention score (Q·K^T)，
     score 最高的 topic 就是最相关的。
     这是最"正宗"的做法，和 transformer 内部机制一致。

 使用示例：
   from topic_relevance import TopicRelevanceScorer

   scorer = TopicRelevanceScorer(method="keyword_match")
   scores = scorer.score(
       query="What is the first topic we discussed?",
       topic_names=["The role of art in society", ...],
   )
   # → [0.95, 0.05, 0.05, ...]   topic 0 分数最高
================================================================================
"""

import re
import torch
import numpy as np


# ================================================================
# 序数词 → 数字映射（给 keyword_match 用）
# ================================================================
ORDINAL_TO_NUMBER = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
}


# ================================================================
# 方案 A: 关键词/规则匹配
# ================================================================
def keyword_match(query, topic_names, topic_labels=None):
    """
    从 query 文本中提取序数词，定位目标 topic。

    适用场景：LongChat 的查询是模板化的，比如
      "What is the first topic we discussed?"
      "What is the third topic we discussed?"
    这种场景下规则匹配准确率 100%，不需要模型。

    参数：
        query:        用户的查询字符串
        topic_names:  15 个 topic 名称的列表（来自 prompt 中的文本）
        topic_labels: 15 个 label 的列表（来自 JSON 的 label 字段）
                      如果不传，就用 topic_names

    返回：
        scores: list[float] — 长度 15，目标 topic 位置为 1.0，其余为 0.0
        method_info: dict — 关于匹配过程的额外信息
    """
    query_lower = query.lower()

    # 尝试匹配序数词
    matched_topic_idx = None
    matched_word = None

    for word, number in ORDINAL_TO_NUMBER.items():
        if word in query_lower:
            matched_topic_idx = number - 1  # 转 0-based index
            matched_word = word
            break

    # 也尝试匹配数字（"topic 3" / "topic number 3"）
    if matched_topic_idx is None:
        num_match = re.search(r'topic\s*(?:number\s*)?(\d+)', query_lower)
        if num_match:
            matched_topic_idx = int(num_match.group(1)) - 1
            matched_word = num_match.group(1)

    # 生成分数
    if matched_topic_idx is not None and 0 <= matched_topic_idx < len(topic_names):
        scores = [0.0] * len(topic_names)
        scores[matched_topic_idx] = 1.0
        method_info = {
            "method": "keyword_match",
            "matched_topic_idx": matched_topic_idx,
            "matched_word": matched_word,
            "topic_name": topic_names[matched_topic_idx],
        }
    else:
        # 没匹配到 → 均匀分数（或者可以 fallback 到其他方法）
        scores = [1.0 / len(topic_names)] * len(topic_names)
        method_info = {
            "method": "keyword_match",
            "matched_topic_idx": None,
            "matched_word": None,
            "note": "no ordinal found in query, returning uniform scores",
        }

    return scores, method_info


# ================================================================
# 方案 B: Embedding 语义相似度
# ================================================================
def embedding_similarity(query, topic_names, model=None, tokenizer=None):
    """
    用文本 embedding 计算 query 和每个 topic name 的语义相似度。

    这个方法不依赖 query 里有明确的序数词，而是看「语义上」query
    更接近哪个 topic。比如 query 问 "what's the role of art"，
    即便没写 "first"，也能匹配到 "The role of art in society"。

    参数：
        query:       用户查询字符串
        topic_names: 15 个 topic 名称
        model:       可选，预加载的 sentence-transformers 模型
        tokenizer:   可选

    返回：
        scores:      list[float] — 长度 15，余弦相似度（已归一化）
        method_info: dict
    """

    # 延迟加载：第一次调用时才加载模型
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            # 如果没有 sentence_transformers，用简单的词重叠做 fallback
            return _word_overlap_fallback(query, topic_names)

    # 编码
    query_emb = model.encode([query], convert_to_tensor=True)
    topic_embs = model.encode(topic_names, convert_to_tensor=True)

    # 余弦相似度
    query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)
    topic_embs = topic_embs / topic_embs.norm(dim=1, keepdim=True)

    similarities = (query_emb @ topic_embs.T).squeeze(0)  # (15,)
    similarities = similarities.cpu().numpy()

    # 归一化到 [0, 1] 区间（类似 softmax）
    scores = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
    scores = scores.tolist()

    method_info = {
        "method": "embedding_similarity",
        "model": "all-MiniLM-L6-v2",
        "raw_similarities": similarities.tolist(),
    }

    return scores, method_info


def _word_overlap_fallback(query, topic_names):
    """简单词重叠作为 fallback（不依赖 sentence_transformers）"""
    query_words = set(query.lower().split())
    scores = []
    for name in topic_names:
        name_words = set(name.lower().split())
        overlap = len(query_words & name_words)
        scores.append(overlap)
    total = sum(scores)
    if total > 0:
        scores = [s / total for s in scores]
    else:
        scores = [1.0 / len(topic_names)] * len(topic_names)

    return scores, {"method": "word_overlap_fallback", "note": "sentence_transformers not available"}


# ================================================================
# 方案 C: Attention Score（Q-K 注意力分数）★ 导师提到的方法
# ================================================================
def attention_score(query_ids, topic_keys, model, layer_idx=-1):
    """
    用模型的 Q-K 注意力分数来判断 query 和每个 topic 的相关性。

    原理：
      Transformer 的 attention 机制本来就是 Q·K^T —— query 向量和
      key 向量做点积，得分高的位置就是「模型认为相关的」位置。
      把某个 topic 的所有 K 和 query 的 Q 做注意力运算，
      取平均 attention score → 该 topic 的「相关性分数」。

    参数：
        query_ids:   query 的 token IDs，shape (1, query_len)
        topic_keys:  list[torch.Tensor] — 15 个 topic 的 K 张量
                     每个 topic_key 的 shape: (num_layers, num_heads, topic_len, head_dim)
                     或者可以是单个 tensor —— 会按 topic token range 从完整 K 中切片
        model:       加载好的 HuggingFace 模型（用于拿 Q 和 attention 权重）
        layer_idx:   用哪一层的 attention。默认 -1（最后一层）。
                    浅层更关注局部语法，深层更关注语义。

    返回：
        scores:      list[float] — 长度 15，归一化后的 attention 分数
        method_info: dict

    注意：
      这个方法需要加载模型 + 做一次前向传播来拿 Q，比方案 A/B 重得多。
      但在学术上最有说服力——它证明了「相关性判断可以依赖模型内部信号」。
    """

    from transformers.modeling_outputs import BaseModelOutputWithPast

    # 1. 把 query_ids 过模型，拿到最后一层（或指定层）的 hidden states
    with torch.no_grad():
        outputs = model.model(
            input_ids=query_ids,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
        )
        # hidden_states[layer] shape: (1, query_len, hidden_dim)
        hidden = outputs.hidden_states[layer_idx]  # (1, query_len, 4096)

    # 2. 拿模型的 Q 投影矩阵
    #    不同模型结构不同，这里以 Llama/Mistral 为例
    num_layers = len(topic_keys[0]) if isinstance(topic_keys[0], (list, tuple)) else len(topic_keys)

    # 拿指定层的 Q 投影权重
    layer = model.model.layers[layer_idx]
    q_proj = layer.self_attn.q_proj  # Linear(4096, 4096)
    num_heads = layer.self_attn.num_heads
    head_dim = layer.self_attn.head_dim
    num_kv_heads = getattr(layer.self_attn, 'num_key_value_heads', num_heads)

    # 3. 算 Q: hidden @ W_Q → reshape 成多头
    Q = q_proj(hidden)  # (1, query_len, 4096)
    Q = Q.view(1, -1, num_heads, head_dim).transpose(1, 2)  # (1, num_heads, query_len, head_dim)

    # 4. 对每个 topic 的 K，算 attention score
    scores = []
    for topic_key_layers in topic_keys:
        # topic_key_layers: (num_layers, num_kv_heads, topic_len, head_dim)
        # 或 list of tensors，每层一个
        if isinstance(topic_key_layers, list):
            K_layer = topic_key_layers[layer_idx]  # (num_kv_heads, topic_len, head_dim)
        else:
            K_layer = topic_key_layers[layer_idx]  # 同上

        # 确保 K 在正确设备上，加 batch 维
        K_layer = K_layer.to(Q.device)
        if K_layer.dim() == 3:
            K_layer = K_layer.unsqueeze(0)  # (1, num_kv_heads, topic_len, head_dim)

        # GQA: K/V heads 可能少于 Q heads，需要扩展到相同数量
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            K_layer = K_layer.repeat_interleave(n_rep, dim=1)  # (1, num_heads, topic_len, head_dim)

        # Q·K^T / sqrt(d_k)
        scale = head_dim ** 0.5
        attn_weights = torch.matmul(Q, K_layer.transpose(-2, -1)) / scale
        # attn_weights shape: (1, num_heads, query_len, topic_len)

        # 取平均（跨 head 和 query token）
        mean_score = attn_weights.mean().item()
        scores.append(mean_score)

    # 5. 归一化
    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    scores = scores.tolist()

    method_info = {
        "method": "attention_score",
        "layer_used": layer_idx,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }

    return scores, method_info


# ================================================================
# 统一接口
# ================================================================
class TopicRelevanceScorer:
    """
    统一的相关性打分接口。

    用法：
        scorer = TopicRelevanceScorer(method="keyword_match")
        scores, info = scorer.score(query="what is the first topic...",
                                     topic_names=[...])

    方法：
        - "keyword_match"      → keyword_match()
        - "embedding"          → embedding_similarity()
        - "attention"          → attention_score()
        - "auto"               → 自动选择：先试 keyword，失败→embedding→attention
    """

    def __init__(self, method="keyword_match", model=None, tokenizer=None):
        """
        参数：
            method: "keyword_match" | "embedding" | "attention" | "auto"
            model: 预加载的模型（embedding 和 attention 方法需要）
            tokenizer: 预加载的 tokenizer
        """
        self.method = method
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = None  # lazy load

    def score(self, query, topic_names, **kwargs):
        """
        对 query 和 15 个 topic 做相关性打分。

        参数：
            query:       用户查询字符串
            topic_names: 15 个 topic 名称的列表
            **kwargs:    传递给具体方法的额外参数
                         - topic_keys: list[tensor] (attention 方法需要)
                         - query_ids: tensor (attention 方法需要)
                         - topic_labels: list[str] (keyword 方法可选)

        返回：
            scores:      list[float] — 长度 15，总和为 1（或接近 1）
            method_info: dict — 打分方法的元信息
        """

        if self.method == "keyword_match":
            return keyword_match(query, topic_names, kwargs.get("topic_labels"))

        elif self.method == "embedding":
            return embedding_similarity(query, topic_names, self.embedding_model)

        elif self.method == "attention":
            if "query_ids" not in kwargs or "topic_keys" not in kwargs:
                raise ValueError("attention 方法需要 query_ids 和 topic_keys 参数")
            return attention_score(
                kwargs["query_ids"],
                kwargs["topic_keys"],
                self.model,
                kwargs.get("layer_idx", -1),
            )

        elif self.method == "auto":
            # 先试 keyword → embedding → attention
            scores, info = keyword_match(query, topic_names, kwargs.get("topic_labels"))
            if info["matched_topic_idx"] is not None:
                return scores, info
            # keyword 没匹配到，用 embedding
            try:
                scores, info = embedding_similarity(query, topic_names, self.embedding_model)
                return scores, info
            except Exception:
                pass
            # 最后试 attention
            if "query_ids" in kwargs and "topic_keys" in kwargs:
                return attention_score(
                    kwargs["query_ids"], kwargs["topic_keys"],
                    self.model, kwargs.get("layer_idx", -1),
                )
            # 全失败 → 返回均匀分数
            n = len(topic_names)
            return [1.0 / n] * n, {"method": "fallback_uniform"}

        else:
            raise ValueError(f"未知方法: {self.method}，可选: keyword_match, embedding, attention, auto")

    def get_best_topic(self, query, topic_names, **kwargs):
        """返回得分最高的 topic 索引和名称"""
        scores, info = self.score(query, topic_names, **kwargs)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, topic_names[best_idx], scores[best_idx], info


# ================================================================
# 测试代码
# ================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from topic_boundary import load_longchat_sample

    # 读数据
    data = load_longchat_sample("data/longchat.jsonl", doc_id=0)
    topic_names = data['label']  # 15 个 topic name

    # LongChat 数据集里 prompt 末尾的 query
    query = "What is the first topic we discussed? Only give me the topic name."

    print("=" * 70)
    print("测试 topic_relevance.py")
    print("=" * 70)
    print(f"Query: \"{query}\"")
    print()

    # --- 测试方案 A: keyword ---
    print("--- 方案 A: keyword_match ---")
    scorer_a = TopicRelevanceScorer(method="keyword_match")
    scores_a, info_a = scorer_a.score(query, topic_names)
    best_idx, best_name, best_score, _ = scorer_a.get_best_topic(query, topic_names)
    print(f"  Best: [{best_idx}] \"{best_name}\" (score={best_score:.4f})")
    print(f"  Info: {info_a}")
    print()

    # --- 测试方案 B: word overlap fallback ---
    print("--- 方案 B: word_overlap（无 sentence_transformers 时的 fallback）---")
    scores_b, info_b = _word_overlap_fallback(query, topic_names)
    best_idx_b = max(range(len(scores_b)), key=lambda i: scores_b[i])
    print(f"  Best: [{best_idx_b}] \"{topic_names[best_idx_b]}\" (score={scores_b[best_idx_b]:.4f})")
    print(f"  All scores: {[f'{s:.2f}' for s in scores_b]}")
    print()

    # --- 测试: 不同 query 的匹配 ---
    print("--- 测试不同 query ---")
    test_queries = [
        "What is the first topic we discussed?",
        "What is the third topic we discussed?",
        "What is the topic 7 we discussed?",
        "what's the 10th topic?",
        "Tell me about the last topic we discussed",   # 这个 keyword 匹配不到
    ]
    for q in test_queries:
        scores, info = scorer_a.score(q, topic_names)
        if info["matched_topic_idx"] is not None:
            print(f"  \"{q}\" → [{info['matched_topic_idx']}] \"{info['topic_name']}\"")
        else:
            print(f"  \"{q}\" → 未匹配（需 fallback）")
