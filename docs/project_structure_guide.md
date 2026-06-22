# Project Structure Guide

这份文件专门回答一个问题：

```text
现在这个工程里，
哪些是主线代码，
哪些是辅助代码，
哪些只是历史遗留可以先放一边。
```

## 一、主线核心

### `qmsum_sim.py`

当前 QMSum 主实验入口。

现在它的角色更像：

```text
总控 / orchestration
```

负责把下面几块串起来：

```text
数据读取
-> 路由
-> 评测
-> summary / trace 输出
```

### `qmsum_data.py`

数据侧核心模块。

负责：

- 读取一个 QMSum 样本
- 拼接 transcript prompt
- 处理 `relevant_text_span`
- 构建 topic nodes

### `qmsum_routing.py`

路由算法核心模块。

负责：

- topic embedding 打分
- hierarchical candidates 构建
- topic -> chunk 两级路由

### `qmsum_eval.py`

评测核心模块。

负责：

- turn recall / precision / F1
- transfer accounting
- summary 汇总与打印

### `qmsum_trace.py`

解释和调试模块。

负责导出单 case markdown / json，帮助你讲清楚：

```text
这个 query 是怎么一步步选到 topic 和 chunk 的
```

### `prepare_qmsum_data.py`

数据预处理入口。

把原始 QMSum 数据整理成主线实验真正使用的：

```text
data/qmsum_structured/{train,val,test}.jsonl
```

### `src/utils.py`

底层通用工具。

负责：

- 模型加载
- tokenizer / KV 处理
- 量化相关辅助逻辑

## 二、主线实验脚本

### `scripts_qmsum/run_qmsum_topic_label_ablation.sh`

当前最值得保留的脚本之一。

它验证：

```text
topic label 是否真的帮助顶层 topic 路由
```

### `scripts_qmsum/run_qmsum_top1_chunk_budget_sweep.sh`

当前最贴近主线的 sweep。

它固定：

```text
top-level = 1 topic
```

然后只扫 chunk budget。

### `_archived_unused_20260619/scripts_qmsum/run_qmsum_hierarchical_sweep.sh`

较早期但仍有参考价值的 QMSum hierarchical sweep。

## 三、辅助理解模块

### `distributed_sim.py`

旧的 ShareGPT / passkey / flat-routing 主实验文件。

它现在不是主线，但仍然重要，因为：

- 你的很多思路是从这里长出来的
- 一些底层函数仍然从这里复用

### `experiment_chunk_split.py`

chunk 级 Q-K 评分辅助模块。

它不是汇报重点，但它是 chunk 路由能工作的底层支持之一。

### `scripts_sharegpt/run_chunk_routing_sweep.sh`

旧 ShareGPT 路线的自动脚本。

可以当背景材料，但不是当前最重要的内容。

### `scripts_qmsum/preview_qmsum_sample.py`

只是看数据样本的小工具，不是算法核心。

## 四、文档

优先级最高的文档：

- `README.md`
- `docs/code_reading_map_cn.md`
- `docs/core_idea.md`
- `docs/progress.md`

其他文档：

- `docs/next_steps_checklist.md`
- `docs/reference_paper_code_notes.md`
- `docs/data_provenance.md`
- `docs/cloud_environment.md`

## 五、历史/降权内容

### `scripts_qmsum/run_qmsum_topic_strategy_compare.sh`

保留作为历史对照。

它记录过：

```text
embedding vs rerank
```

但现在不属于最该优先跑的主线脚本。

### `_archived_unused_20260605/`
### `_archived_unused_20260606/`

这些都属于已降权或历史实验内容。

保留是为了：

- 防止以后回查
- 防止误删旧结果

但它们不应再被当作当前代码主线。

## 六、推荐阅读顺序

如果你要重新理解整个工程，最推荐这样读：

1. `README.md`
2. `docs/code_reading_map_cn.md`
3. `docs/core_idea.md`
4. `qmsum_sim.py`
5. `qmsum_data.py`
6. `qmsum_routing.py`
7. `qmsum_eval.py`
8. `logs/qmsum_trace/doc_5_query_0.md`

如果你只是补旧背景，再去看：

1. `distributed_sim.py`
2. `scripts_sharegpt/run_chunk_routing_sweep.sh`
