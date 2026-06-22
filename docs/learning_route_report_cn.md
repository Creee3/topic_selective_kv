# Topic-Selective KV 工程学习与汇报记录

> 这份文档按学习路线记录当前工程。Markdown 预览工具可以根据标题自动生成目录。

## 第 0 章：当前工程主线总览

### 0.1 当前工程一句话

当前工程可以概括为：

```text
一个面向 distributed KV cache 的 query-aware selective KV transfer prototype。
```

中文解释是：

```text
一个根据 query 选择性传输 KV cache 的原型系统。
```

它不是普通 QMSum 摘要系统，也不是普通 RAG/retrieval 系统。它用 QMSum 会议数据模拟长上下文，用 topic nodes 模拟语义分片，用 virtual nodes 模拟分布式 KV 节点，用 lexical routing 先选 topic，再用模型内部 Q-K attention affinity 在 topic 内选 chunk，最后评估少传 KV 后的证据召回、答案质量、KV saving 和 TTFT。

### 0.2 为什么要做这件事

长上下文 LLM 对一整段历史内容做过 prefill 后，会产生很大的 KV cache。

如果这些 KV cache 分布在多个远端节点上，每次新 query 到来都 fetch 全量 KV，会带来大量不必要传输。因此当前工程研究：

```text
能不能根据 query 判断哪些 topic/chunk 的 KV 可能有用，
只 fetch 这些 KV，
从而减少 KV 传输量和 TTFT，
同时尽量不损害 evidence recall 和 answer quality。
```

### 0.3 当前代码主入口

当前主线配置在：

```text
qmsum_mainline_config.py
```

里面最重要的是：

```text
MAINLINE_PROFILES["current"]
```

当前主线运行入口是：

```text
scripts_qmsum/run_qmsum_current_mainline.sh
```

这个脚本本质上调用：

```bash
python qmsum_mainline.py --mainline_profile current
```

理解工程时，优先看：

```text
docs/current_mainline_cn.md
qmsum_mainline_config.py
qmsum_mainline.py
qmsum_mainline_routing.py
qmsum_eval.py
qmsum_data.py
scripts_qmsum/run_qmsum_current_mainline.sh
```

其他旧脚本更多是 ablation 或历史实验。

### 0.4 当前主线完整流程

当前主线可以表示为：

```text
QMSum meeting
  ↓
构造 prompt_text 和 turn_boundaries
  ↓
本地跑一次 full prefill，拿到完整 KV cache
  ↓
用 QMSum topic_list / 人工标注构造 topic nodes
  ↓
把 topic nodes 打包到 virtual nodes
  ↓
query 到来
  ↓
lexical/BM25 粗路由，选 top-1 topic
  ↓
candidate prefilter / dynamic pool / coarse segment gate
  ↓
在剩余候选 chunks 上做 exact Q-K scoring
  ↓
选出 top-k KV chunks
  ↓
计算 selected turns / transfer segments / KV bytes / TTFT
  ↓
可选：生成 full answer、selected answer、oracle answer，比较 F1
```

可以压缩成：

```text
数据准备
-> 粗路由
-> 中间裁剪
-> Q-K 细路由
-> 传输评估
-> 答案评估
```

### 0.5 一个具体例子

假设 QMSum 里有一场会议：

```text
meeting = train_00005
```

里面有很多 turns：

```text
turn 0: Project Manager says ...
turn 1: User Interface says ...
turn 2: Marketing says ...
...
turn 180: ...
```

还有 topic_list：

```text
topic 0: remote colour and logo
topic 1: battery design
topic 2: user interface
topic 3: cost and production
```

现在来了一个 query：

```text
What did the team decide about the remote control colour?
```

系统不会直接 fetch 整场会议所有 KV，而是先做 coarse routing：

```text
query 里有 remote / colour
topic 0 里也有 remote / colour / logo
```

所以粗路由可能选中：

```text
topic 0
```

然后系统只在 topic 0 对应的 turns/chunks 里做 Q-K scoring。

例如 topic 0 里有：

```text
chunk A: turn 15 的一段
chunk B: turn 16 的一段
chunk C: turn 35 的一段
chunk D: turn 40 的一段
```

Q-K scoring 判断：

```text
当前 query 的 Q
和每个历史 chunk 的 K
在模型 attention 空间里有多对齐
```

最后可能选：

```text
chunk A
chunk C
```

于是传输成本只按 selected KV 估算：

```text
selected KV = chunk A KV + chunk C KV
```

而不是：

```text
full KV = 整场会议所有 tokens 的 KV
```

这就是 selective KV transfer。

### 0.6 topic / node / virtual node 的当前理解

`topic node` 是语义节点。

例如：

```text
topic 0 = remote colour and logo
topic 1 = battery design
topic 2 = user interface
```

这些 topic node 可以来自：

```text
QMSum topic_list
人工标注
业务系统结构
离线 topic 标注器
```

老师已经说 topic node 可以人工标注，所以当前工程不需要把“自动发现 topic”作为核心贡献。

`deployable node` 是真实系统里的 KV 存储节点。

例如：

```text
node 0 存 topic 0、topic 1 的 KV
node 1 存 topic 2 的 KV
node 2 存 topic 3 的 KV
```

`virtual node` 是当前代码里模拟 deployable node 的对象。

因为目前没有真实多机部署，所以代码用 virtual node 模拟：

```text
如果 topic 被放到不同节点上，selective fetch 会传多少 KV？
会访问几个 node？
会形成多少 transfer segment？
```

当前代码支持三种 topic 到 virtual node 的放置方式：

```text
contiguous   按 topic 顺序连续打包
round_robin  轮询分配到不同 node
manual       从 JSON 文件读取人工 topic->node 映射
```

manual layout 示例：

```text
docs/manual_topic_node_layout_example.json
```

运行方式示例：

```bash
python qmsum_mainline.py \
  --mainline_profile current \
  --node_assignment_mode manual \
  --topic_node_layout_path docs/manual_topic_node_layout_example.json
```

### 0.7 current profile 固化的主线策略

当前 `current` profile 里固定了这些核心策略：

```text
hier_top_topics = 1
route_chunk_size = 128
route_top_k = 12
route_per_head = True
route_neighbor_expand = 0
```

含义是：

```text
粗路由只选 1 个 topic；
每个 chunk 大小 128 tokens；
细路由选 top-k chunks；
Q-K scoring 使用 per-head 思路；
不做 neighbor expansion。
```

为什么不做 neighbor expansion？

因为最新结果显示：

```text
neighbor_expand=1 没有提升 selected F1 / turn recall
但增加 KV 和 TTFT
```

所以当前主线保持：

```text
route_neighbor_expand=0
```

current profile 还包含系统侧优化：

```text
candidate prefilter
dynamic candidate pool
coarse segment gate
batched exact Q-K
candidate key cache
```

这些说明工程已经从：

```text
能不能选到相关 chunk
```

推进到：

```text
能不能更便宜地选到相关 chunk
```

### 0.8 当前工程边界

当前工程不是完整 distributed serving system。

它现在不是：

```text
真的实现了多机 KV cache serving engine
```

而是：

```text
先在本地得到 full KV cache，
然后切片模拟 selective fetch。
```

真实系统中，KV 一开始就分布在多个机器上；当前 prototype 中，先本地算出完整 KV，再根据 topic/chunk/node 映射模拟如果分布式 fetch 会怎样。

因此汇报时应该说：

```text
当前系统是 simulation/prototype，
核心验证 selective KV transfer 的 routing decision 和 cost-quality trade-off。
```

不应该说：

```text
我已经实现完整分布式推理系统。
```

### 0.9 给老师汇报的开场版本

可以这样汇报：

```text
老师，我现在把工程主线重新收敛了一下。

当前工作不是做自动 topic discovery，而是假设 topic nodes 可以由人工或数据集标注给定。
在这个前提下，我研究的是 distributed KV cache 场景下，query 到来后如何只选择性传输有用的 KV。

具体流程是：
先用 QMSum meeting 构造长上下文和 KV cache；
再把 topic nodes 映射到 virtual nodes；
query 先通过 lexical/BM25 选中相关 topic；
然后在 topic 内部通过模型自身的 Q-K attention affinity 给 chunk 打分；
最后只 fetch selected chunks 的 KV，并评估 evidence recall、answer F1、KV saving 和 TTFT。
```

### 0.10 本章核心记忆

当前工程主线是：

```text
给定语义 topic nodes
-> 映射到 virtual/deployable KV nodes
-> query-aware coarse routing
-> cheap candidate pruning
-> exact Q-K fine chunk routing
-> selective KV transfer accounting
-> quality/cost evaluation
```

## 第 1 章：研究问题

### 1.1 为什么会有 distributed KV cache

待补充。

### 1.2 为什么 full KV fetch 浪费

待补充。

### 1.3 当前工程的问题定义

待补充。

## 第 2 章：LLM KV cache 原理

### 2.1 KV cache 是什么

待补充。

### 2.2 prefill 在做什么

待补充。

### 2.3 decode 在做什么

待补充。

## 第 3 章：Q-K attention 原理

### 3.1 Q / K / V 分别是什么

在 Transformer attention 里，每个 token 进入某一层后，会被线性投影成三类向量：

```text
Q = Query
K = Key
V = Value
```

可以先用一句话理解：

```text
Q 表示“我现在想找什么”
K 表示“我这里有什么特征可以被别人匹配”
V 表示“如果别人关注我，我实际提供什么信息”
```

更具体一点：

```text
当前 query token 会产生 Q。
历史上下文 token 已经在 prefill 阶段产生 K/V，并保存在 KV cache 里。
decode 或后续 query 需要利用历史上下文时，就用当前 Q 去和历史 K 做匹配。
匹配分数高的历史 token，其 V 会被更多读取。
```

标准 attention 公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

其中：

```text
QK^T / sqrt(d)
```

就是 attention score，也可以理解为：

```text
当前 query 对历史 token 的注意力亲和度。
```

你的工程主要借用的是这个公式里的前半段：

```text
QK^T / sqrt(d)
```

也就是：

```text
先不急着乘 V，先看 query 的 Q 和历史 chunk 的 K 到底有多匹配。
```

这就是 Q-K routing 的核心。

### 3.2 为什么 query 的 Q 可以给历史 chunk 的 K 打分

因为 LLM 本来就是用 Q 和 K 来决定：

```text
当前 token 应该关注历史里的哪些 token。
```

你的方法只是把这个机制拿出来当 routing signal。

普通 decode 时，模型内部会做：

```text
当前 token 的 Q
和所有历史 token 的 K
计算 attention affinity
```

你的 selective KV routing 做的是：

```text
当前 query 的 Q
和每个历史 chunk 的 K
计算 attention affinity
```

然后把每个 chunk 的 affinity 聚合成一个分数：

```text
chunk score
```

分数越高，表示：

```text
模型内部 attention 空间认为这个 chunk 更值得被 query 关注。
```

所以它不是额外发明了一个相关性定义，而是在问：

```text
如果模型真的看这段历史，它的 attention 机制会更倾向于看哪里？
```

这就是为什么 Q-K 可以用于 chunk routing。

#### 3.2.1 这和普通 retrieval 有什么不同

普通 retrieval 通常是：

```text
query text -> embedding
chunk text -> embedding
比较两个 embedding 的语义相似度
```

它用的是外部检索器或 embedding 模型的语义空间。

你的 Q-K routing 是：

```text
query text 在 LLM 某一层里的 Q
历史 chunk 在同一个 LLM KV cache 里的 K
比较 Q 和 K 的 attention affinity
```

它用的是 LLM 自己 attention 层里的空间。

两者区别可以写成：

```text
普通 retrieval:
  text/query embedding space

Q-K routing:
  model internal attention space
```

所以你的方法更贴近 KV cache 场景，因为你不是在问：

```text
这段文本语义上像不像 query？
```

而是在问：

```text
如果把这段 KV 给模型，模型自己的 attention 会不会想看它？
```

这就是你和普通 RAG/retrieval 的区别。

### 3.3 layer / head / token 维度例子

#### 3.3.1 当前模型里的真实维度

你现在用的 Mistral 类模型大概是：

```text
hidden_size = 4096
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
```

为什么：

```text
head_dim = 128
```

因为：

```text
hidden_size / num_attention_heads
= 4096 / 32
= 128
```

注意这里有一个重要点：

```text
num_attention_heads = 32
num_key_value_heads = 8
```

这说明模型使用了 GQA，也就是 grouped-query attention。

含义是：

```text
Q 有 32 个 heads
K/V 只有 8 个 KV heads
多个 Q heads 共享一组 K/V heads
```

所以代码里会看到：

```python
if num_kv_heads != num_heads:
    n_rep = num_heads // num_kv_heads
    K_layer = K_layer.repeat_interleave(n_rep, dim=1)
```

意思是：

```text
把 8 个 KV heads 扩展成 32 个 heads，
这样才能和 32 个 Q heads 对齐做矩阵乘法。
```

#### 3.3.2 代码和公式怎么对应

核心代码在 `experiment_chunk_split.py` 的 `_get_qk_attention(...)`。

关键逻辑是：

```python
hidden = outputs.hidden_states[layer_idx]
Q = q_proj(hidden)
Q = Q.view(1, -1, num_heads, head_dim).transpose(1, 2)

K_layer = ck[layer_idx].unsqueeze(0).to(Q.device)
if num_kv_heads != num_heads:
    n_rep = num_heads // num_kv_heads
    K_layer = K_layer.repeat_interleave(n_rep, dim=1)

attn = torch.matmul(Q, K_layer.transpose(-2, -1)) / scale
per_head = attn.mean(dim=(2, 3)).squeeze(0)
```

逐行对应公式：

```text
hidden = outputs.hidden_states[layer_idx]
```

表示：

```text
拿到 query token 在某一层的输入 hidden states。
```

```text
Q = q_proj(hidden)
```

对应：

```text
Q = hidden × W_Q
```

也就是把 hidden state 投影成 Query 向量。

```text
K_layer = ck[layer_idx]
```

表示：

```text
拿出某个历史 chunk 在同一层里的 Key。
```

```text
torch.matmul(Q, K_layer.transpose(-2, -1))
```

对应：

```text
QK^T
```

```text
/ scale
```

对应：

```text
/ sqrt(head_dim)
```

也就是：

```text
QK^T / sqrt(d)
```

最后：

```text
per_head = attn.mean(dim=(2, 3))
```

表示：

```text
把 query_len × chunk_len 这个矩阵平均成每个 head 一个分数。
```

#### 3.3.3 affinity 分数是什么

代码里 `attn` 的形状是：

```text
(1, num_heads, query_len, chunk_len)
```

在当前模型里可以理解成：

```text
(1, 32, query_len, chunk_len)
```

它的含义是：

```text
batch 里的第 1 个样本，
32 个 attention heads 里，
query 的每个 token，
对 chunk 的每个 token，
都有一个 Q-K 匹配分数。
```

也就是说，对于某个 head：

```text
attn[head, i, j]
```

表示：

```text
query 第 i 个 token 的 Q
和 chunk 第 j 个 token 的 K
有多对齐。
```

这个分数就叫 affinity，中文可以理解成：

```text
亲和度
匹配度
注意力倾向
```

分数越高，表示：

```text
在这个 head 的 attention 空间里，
query token 越倾向于关注这个历史 token。
```

#### 3.3.4 一个 2 维玩具例子

真实工程里 head_dim 是 128 维。为了看懂，我们先用 2 维例子。

假设 query 的 Q 是：

```text
Q = [1, 2]
```

chunk A 的某个 token 的 K 是：

```text
K_A = [2, 1]
```

点积：

```text
Q · K_A
= 1*2 + 2*1
= 4
```

chunk B 的某个 token 的 K 是：

```text
K_B = [-1, 0]
```

点积：

```text
Q · K_B
= 1*(-1) + 2*0
= -1
```

所以：

```text
chunk A token 的匹配分数更高。
```

直觉是：

```text
Q 和 K_A 指向相似方向；
Q 和 K_B 方向不一致。
```

真实模型里不是 2 维，而是：

```text
head_dim = 128
```

所以实际点积是：

```text
Q · K
= Q[0]*K[0] + Q[1]*K[1] + ... + Q[127]*K[127]
```

但本质仍然是：

```text
Q 和 K 越对齐，分数越高。
```

#### 3.3.5 展开到 token 矩阵

上面的例子只有一个 query token 和一个 chunk token。

真实情况是：

```text
query 有多个 tokens
chunk 也有多个 tokens
```

假设：

```text
query_len = 3
chunk_len = 4
```

那么一个 head 里会得到一个矩阵：

```text
          chunk token 0   chunk token 1   chunk token 2   chunk token 3
query 0        0.2             1.5            -0.3             0.7
query 1        2.1             0.4             0.8            -0.2
query 2       -0.5             0.9             1.2             0.3
```

这个矩阵就是：

```text
query 每个 token 对 chunk 每个 token 的 Q-K affinity。
```

代码中：

```python
attn = torch.matmul(Q, K_layer.transpose(-2, -1)) / scale
```

得到的就是这种矩阵，只不过它同时包含：

```text
batch 维度
head 维度
query token 维度
chunk token 维度
```

所以形状是：

```text
(1, 32, query_len, chunk_len)
```

#### 3.3.6 展开到 head

模型不是只用一个 attention head。

当前模型大概有：

```text
32 个 attention heads
```

每个 head 都有自己的 Q-K 空间。

可以理解成：

```text
head 0 可能关注人名
head 1 可能关注时间
head 2 可能关注决策词
head 3 可能关注指代关系
...
```

这只是直觉解释，不代表每个 head 一定有明确人工语义。

代码里：

```text
attn: (1, 32, query_len, chunk_len)
```

然后：

```python
per_head = attn.mean(dim=(2, 3)).squeeze(0)
```

会得到：

```text
per_head: (32,)
```

也就是：

```text
这个 chunk 在 32 个 heads 上各有一个分数。
```

例子：

```text
chunk A:
  head 0 score = 0.8
  head 1 score = 0.2
  head 2 score = 1.1
  ...
  head 31 score = 0.4
```

这说明：

```text
不同 head 可以从不同角度判断 chunk 是否相关。
```

这也是你之前为什么关注 `route_per_head=True`：

```text
如果只把所有 head 平均，可能会抹掉某些 head 的强信号；
per-head 选择可以保留不同 head 发现的候选 chunk。
```

#### 3.3.7 展开到 layer

模型也不是只有一层。

当前 Mistral 类模型有多层 Transformer，例如 32 层左右。

你的代码默认不会只看一层，而是选择多层：

```text
scoring_layers = [0, 8, 16, 24, last_layer]
```

也就是：

```text
浅层
中层
深层
最后一层
```

为什么要多层？

因为不同层的 Q-K 信号可能不同：

```text
浅层可能更偏局部词形/短语匹配；
中层可能有更强结构信息；
深层可能更接近任务语义。
```

代码里：

```python
all_scores = []
for lidx in layer_indices:
    scores = _get_qk_attention(query_ids, chunk_keys_list, model, lidx)
    all_scores.append(scores)

avg_scores = np.mean(all_scores, axis=0)
```

含义是：

```text
每一层都算一次 chunk × head 分数，
最后对多个 layer 求平均。
```

如果有：

```text
n_chunks = 50
n_heads = 32
n_layers_used = 5
```

那么每层先得到：

```text
(50, 32)
```

5 层平均后仍然是：

```text
(50, 32)
```

也就是：

```text
每个 chunk 在每个 head 上有一个跨层平均后的分数。
```

#### 3.3.8 从 head/layer 分数到 chunk scalar

前面得到的是：

```text
chunk × head
```

但最终排序 chunk 时，需要每个 chunk 一个分数。

这个最终的单个分数，我们之前叫：

```text
chunk scalar
```

scalar 的意思就是：

```text
一个普通数字。
```

例如：

```text
chunk A score = 0.83
chunk B score = 0.41
chunk C score = 1.27
```

代码里在 `qmsum_mainline_routing.py` 的 `score_candidates_exact_qk_batched(...)` 里做：

```python
scores_matrix, _ = qk_score_fn(query_ids, chunk_keys_list, model, scoring_layers)
...
scalar = aggregate_qk_scores(scores, qk_aggregation, qk_topk)
record["score"] = float(scalar)
```

其中：

```text
scores
```

是某个 chunk 的多 head 分数。

```text
aggregate_qk_scores(...)
```

把它聚合成一个 scalar。

常见聚合方式包括：

```text
mean       所有 head 平均
max        取最强 head
topk_mean  取 top-k 强 head 平均
```

最后系统按这个 scalar 排序：

```text
score 越高，chunk 越优先被选中。
```

#### 3.3.9 这一章和你的工程主线怎么连起来

Q-K attention 原理在你的工程里承担的是：

```text
fine routing signal
```

也就是：

```text
粗路由先用 lexical/BM25 选 topic；
进入 topic 后，再用 Q-K 判断哪个 chunk 更值得 fetch。
```

所以它不是替代全部 routing。

当前主线不是：

```text
直接用 Q-K 从所有 topic 里找答案。
```

而是：

```text
coarse lexical topic routing
-> topic-local Q-K chunk routing
```

为什么这样设计？

因为之前实验已经说明：

```text
粗粒度 topic/node 级 Q-K 不稳定；
但 topic 内部的 chunk 级 Q-K 更适合做细粒度证据选择。
```

因此 Q-K 在你的论文里应该这样讲：

```text
Q-K is used as a model-internal fine-grained KV chunk relevance signal,
not as the only global router.
```

中文就是：

```text
Q-K 是模型内部的细粒度 KV chunk 相关性信号，
不是唯一的全局粗路由器。
```

#### 3.3.10 本章核心记忆

这一章最重要的是下面几句话：

```text
1. Q 表示当前 query 想找什么。
2. K 表示历史 token 提供什么可匹配特征。
3. QK^T / sqrt(d) 表示 query token 对历史 token 的 attention affinity。
4. 你的 Q-K routing 用这个 affinity 给历史 chunk 打分。
5. 真实工程里会跨 query tokens、chunk tokens、heads、layers 做聚合。
6. 最终每个 chunk 得到一个 scalar score，用来排序和选择。
7. Q-K 在当前系统里主要用于 topic 内 fine routing，而不是全局 topic routing。
```

## 第 4 章：系统实体

### 4.1 topic node

待补充。

### 4.2 deployable node

待补充。

### 4.3 virtual node

待补充。

### 4.4 chunk

待补充。

### 4.5 transfer segment

待补充。

### 4.6 topic node 到 deployable node 的构造

待补充。

## 第 5 章：QMSum 数据结构

### 5.1 meeting

待补充。

### 5.2 transcript / turn

待补充。

### 5.3 topic_list

待补充。

### 5.4 specific_query

待补充。

### 5.5 relevant_text_span

待补充。

## 第 6 章：数据到 token 坐标

### 6.1 为什么 turn_boundaries 是地基

待补充。

### 6.2 char span 到 token span

待补充。

### 6.3 tokenizer offset_mapping 和 add_special_tokens=False

待补充。

## 第 7 章：粗路由

### 7.1 为什么粗路由先选 topic

待补充。

### 7.2 为什么当前用 lexical / BM25

待补充。

### 7.3 representative turns 怎么进入 topic document

待补充。

## 第 8 章：中间候选裁剪层

### 8.1 candidate prefilter

待补充。

### 8.2 dynamic candidate pool

待补充。

### 8.3 coarse segment gate

待补充。

### 8.4 为什么这一层是系统侧优化

待补充。

## 第 9 章：细路由

### 9.1 topic 内 Q-K chunk routing

待补充。

### 9.2 chunk scalar 分数

待补充。

### 9.3 head 聚合与 layer 聚合

待补充。

## 第 10 章：评价指标

### 10.1 selected_turn recall / precision / F1

待补充。

### 10.2 answer F1

待补充。

### 10.3 KV saving

待补充。

### 10.4 TTFT saving

待补充。

### 10.5 传输时延组成

待补充。

## 第 11 章：当前 mainline profile 和默认设置

### 11.1 current / manual / simple profile

待补充。

### 11.2 current profile 的关键参数

待补充。

### 11.3 neighbor expansion 为什么不再默认

待补充。

## 第 12 章：工程边界和论文贡献

### 12.1 当前是 simulation / prototype

待补充。

### 12.2 不应该声称什么

待补充。

### 12.3 应该怎样讲贡献

待补充。

## 第 13 章：下一步突破方向

### 13.1 topic-to-node placement

待补充。

### 13.2 离线 K-summary / descriptor

待补充。

### 13.3 routing-aware transfer layout

待补充。

### 13.4 更真实的 distributed serving 对比

待补充。
