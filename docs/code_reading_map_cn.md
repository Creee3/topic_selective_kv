# 代码阅读地图（中文）

这份文件的目标只有一个：

```text
让你快速知道：
1. 现在主要看哪些代码
2. 每个文件在整个数据流里负责什么
3. 哪些文件只是辅助，暂时可以先不看
```

## 一句话先说清

你现在在做的主线不是“最终问答生成”，而是：

```text
模拟一个分布式 KV / 证据路由系统：
给定一个 query
-> 先判断应该去哪个 topic
-> 再在这个 topic 里面找哪些 chunk 最值得拉取
-> 最后用 QMSum 的标注答案区间，检查你找得准不准
```

## 现在最重要的 5 个文件

### 1. `qmsum_sim.py`

这是总控文件。

你可以把它理解成：

```text
实验导演
```

它本身现在尽量不负责太多细节，而是负责：

- 读参数
- 逐个 doc / query 跑
- 调用数据处理
- 调用路由
- 调用评测
- 最后打印 summary / 保存 json / 保存 trace

如果你只想抓主流程，先看这个。

### 2. `qmsum_data.py`

这是数据整理模块。

它负责：

- `load_qmsum_sample`
  从 `train.jsonl` 里取出第 `doc_id` 个 meeting
- `build_qmsum_prompt`
  把整场会议 transcript 拼成一个长 prompt，并记录每个 turn 对应的 token 范围
- `spans_to_turn_set`
  把 QMSum 里的 `relevant_text_span` 变成 turn 编号集合
- `build_topic_nodes`
  把 QMSum 的 `topic_list` 变成“topic 节点”

你之前总在问“topic 从哪来”，主要就看这个文件。

### 3. `qmsum_routing.py`

这是最核心的算法文件。

它负责两级路由：

```text
topic 级粗路由
-> chunk 级细路由
```

最重要的函数有：

- `score_topics_embedding_qmsum`
  用 embedding 给 topic 打分
- `build_hierarchical_candidates`
  把 turn 切成 chunk 候选块
- `score_hierarchical_topic_chunk`
  先选 topic，再在 topic 内做 Q-K chunk 选择

如果你想回答老师“算法具体怎么实现”，这个文件是主战场。

### 4. `qmsum_eval.py`

这是评测文件。

它负责：

- `compute_selected_turn_metrics`
  算 selected turn 和 GT turn 的 recall / precision / F1
- `build_transfer_accounting`
  算如果真的传输 KV，需要传多少 chunk / topic / segment
- `build_summary_payload`
  汇总所有样本结果
- `print_summary`
  打印你现在看到的那种 summary

你之前一直问“为什么不用直接看生成 F1”，这个文件对应的是：

```text
我们现在验证的是“证据路由对不对”
不是最终生成答案对不对
```

### 5. `qmsum_trace.py`

这是解释型辅助文件。

它负责把单个 case 导出成 markdown / json。

比如你现在经常看的：

```text
logs/qmsum_trace/doc_5_query_0.md
```

它不是主算法，但对“讲清楚过程”特别重要。

## 一条完整数据流

你现在可以把主流程记成下面这 8 步：

```text
1. 从 QMSum 取一个 meeting
2. 取 meeting 里的一个 specific query
3. 把整场 meeting transcript 拼成 prompt
4. 跑模型，拿到整场 meeting 的 KV cache
5. 用 topic_list 建 topic 节点
6. 用 embedding 选最相关的 top topic
7. 只在这个 topic 里面，用 Q-K 分数选 top chunks
8. 用 relevant_text_span 检查这些 chunks 是否覆盖了正确证据
```

## 你应该怎么读代码

推荐顺序：

1. `docs/core_idea.md`
2. `qmsum_sim.py`
3. `qmsum_data.py`
4. `qmsum_routing.py`
5. `qmsum_eval.py`
6. `logs/qmsum_trace/doc_5_query_0.md`

这个顺序比直接从 `qmsum_sim.py` 一路读到底更容易懂。

## 哪些文件是“辅助的”

### `prepare_qmsum_data.py`

只在准备数据时用。

它把原始 QMSum 处理成：

```text
data/qmsum_structured/train.jsonl
data/qmsum_structured/val.jsonl
data/qmsum_structured/test.jsonl
```

如果你现在是在理解主算法，可以先不细看。

### `scripts_qmsum/*.sh`

这些是自动跑实验的脚本。

它们不是算法本身，只是帮你批量运行不同参数。

### `distributed_sim.py`

这是旧主线，主要是 ShareGPT / passkey / flat routing 的模拟器。

它仍然重要，因为：

- 当前很多底层函数还是复用了它
- 你的 QMSum 主线是从它演化来的

但它现在不是你最该优先汇报的文件。

## 哪些文件可以先忽略

### `_archived_unused_20260605/`
### `_archived_unused_20260606/`

这些都属于历史实验分支。

保留它们是为了防止以后要回滚、查旧结果，但当前主线不用看。

## 你汇报时可以怎么说

你现在可以用这段很稳地概括：

```text
我现在在做的是一个分布式 KV 路由仿真。
在 QMSum 上，我把一场长会议先切成 topic 节点，
先用 embedding 做粗粒度 topic 选择，
再在选中的 topic 内部用 Q-K 分数做细粒度 chunk 选择。
最后不是直接看生成答案，而是先看：
选中的 topic / turn / chunk 是否覆盖了 QMSum 标注的相关证据区间。
同时我还统计如果真的传输这些 KV，传输单元数和连续 segment 数会是多少。
```

## 目前你最该记住的主线结论

```text
embedding 更适合做顶层 topic 路由
Q-K 更适合做 topic 内部的细粒度 chunk 选择
```

这就是你现在这条线最核心的思想。

## 2026-06-12 之后怎么读主线

如果你现在是为了理解“当前真正运行的主线代码”，优先顺序改成：

1. `docs/core_idea.md`
2. `docs/progress.md`
3. `qmsum_mainline.py`
4. `qmsum_data.py`
5. `qmsum_mainline_routing.py`
6. `qmsum_eval.py`
7. `qmsum_answering.py`
8. `outputs/qmsum_answer_log_N4_10_20_mainline_lexical_top1_chunks_12_answers.md`

补充说明：

```text
旧的 qmsum_sim.py / qmsum_routing.py 仍然保留，
但它们现在更偏“历史实验总控 / 多策略对比”。

当前冻结主线已经收敛成：
  lexical coarse topic routing
  -> top-1 topic
  -> in-topic Q-K fine chunk routing
  -> answer evaluation
```

主线里一个最近的重要清理是：

```text
coarse topic 文档不再使用 "Topic label: xxx" 这样的前缀，
而是直接使用纯 topic label 文本。
这样不会再平白引入 topic / label 这类无意义 token。
```

还有一个最近的重要可读性改进是：

```text
主线 answer markdown / jsonl 日志现在应该直接暴露 selected chunks，
包括：
  turn
  chunk 编号
  score
  token 范围
  chunk 对应的真实文本
```

所以你以后读主线输出时，不应该只看：

```text
selected_turns / matched_turns / answer F1
```

还应该直接看：

```text
被选中的 chunk 文本到底在讲什么
```
