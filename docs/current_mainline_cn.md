# QMSum 当前主线说明

这份文档只解释当前应该继续推进的主线，不把历史 ablation 全部混进来。

## 1. 研究假设

当前工程不再把“自动发现 topic/node”当作核心问题。

老师已经确认：topic node 可以人工标注。因此在代码和论文表述里，可以把它当作给定输入：

```text
meeting transcript
-> 手工标注或数据集给定的 topic nodes
-> turn 属于哪些 topic nodes
```

QMSum 里暂时用 `topic_list` 模拟这个人工标注过程。真实部署时，`topic_list` 可以换成运维人员、业务系统、会议结构化工具或离线标注器给出的语义节点。

所以当前问题不是：

```text
模型能不能自己发现所有 topic？
```

而是：

```text
给定可部署的语义 topic nodes 后，
query 到来时能不能只 fetch 少量有用 KV，
同时尽量保持证据召回和答案质量？
```

## 2. 当前主线流水线

代码里的当前主线 profile 是：

```text
--mainline_profile current
```

它对应：

```text
QMSum meeting
-> build prompt and turn_boundaries
-> full local prefill once for simulation
-> QMSum/manual labels become topic nodes
-> topics are packed into virtual nodes
-> lexical/BM25 coarse routing selects top-1 topic
-> lexical candidate prefilter keeps a cheaper candidate pool
-> lexical coarse segment gate removes weak local segments
-> batched exact Q-K scores surviving chunks
-> selected chunks are charged to virtual nodes
-> estimate KV bytes, fetch latency, and TTFT
-> optionally generate full/selected/oracle answers and compare F1
```

入口脚本：

```bash
bash scripts_qmsum/run_qmsum_current_mainline.sh
```

核心代码：

```text
qmsum_mainline_config.py   current/simple/manual profiles
qmsum_mainline.py          experiment controller
qmsum_mainline_routing.py  topic routing + chunk routing + virtual-node layout
qmsum_eval.py              selected-turn metrics + transfer/TTFT accounting
qmsum_output.py            TSV/JSONL/Markdown output
```

## 3. Topic、Node、Virtual Node 怎么区分

### topic node

`topic node` 是语义节点。

在 QMSum 里来自：

```text
meeting["topic_list"]
```

例如：

```text
topic 0: remote colour and logo
topic 1: battery and casing
topic 2: user interface
```

这一步现在可以认为是人工给定的，不是你的主要贡献点。

### deployable node

`deployable node` 是系统部署里的存储/服务节点。

论文里可以说：

```text
semantic topic nodes are packed onto deployable KV nodes
```

也就是说，topic 是语义分片，node 是真实或模拟的机器/存储位置。

### virtual node

当前没有真实多机 KV store，所以代码用 `virtual node` 模拟 deployable node。

例子：

```text
topic 0, topic 1 -> virtual node 0
topic 2          -> virtual node 1
topic 3, topic 4 -> virtual node 2
```

如果 query 最终选中了 topic 2 的 chunk，transfer accounting 会记录：

```text
transfer_topic_id = 2
transfer_node_id  = 1
```

这样就能估算：

```text
要访问几个 node
要传多少 KV tokens
要形成多少 transfer segments
大概 TTFT 是多少
```

当前支持三种 topic 到 virtual node 的放置方式：

```text
contiguous   按 topic 顺序连续打包，默认值
round_robin  轮询分配到不同 node
manual       从 JSON 文件读取人工 topic->node 映射
```

manual layout 示例：

```text
docs/manual_topic_node_layout_example.json
```

运行时可以写：

```bash
python qmsum_mainline.py \
  --mainline_profile current \
  --node_assignment_mode manual \
  --topic_node_layout_path docs/manual_topic_node_layout_example.json
```

这一步就是从“QMSum labels 作为 topic nodes”往“可部署节点构造”推进的代码入口。

## 4. 当前 profile 固化的默认设置

`current` profile 集中写在 `qmsum_mainline_config.py`。

关键参数：

```text
hier_top_topics=1
route_chunk_size=128
route_top_k=12
route_per_head=True
route_neighbor_expand=0

dynamic_route_budget=True
dynamic_summary_top_k=16
dynamic_detail_top_k=12
dynamic_balanced_top_k=12

route_candidate_prefilter=lexical
route_candidate_prefilter_factor=6
route_candidate_prefilter_min_keep=48
route_candidate_prefilter_max_keep=128

route_coarse_segment_gate=lexical
route_coarse_segment_size=4
route_coarse_segment_keep_ratio=0.65
route_coarse_segment_min_keep=64

qk_score_batch_size=64
cache_candidate_keys=True

answer_evidence_order=qk_then_time
selected_answer_context_mode=turns
answer_prompt_style=grounded
answer_max_new_tokens=96
```

为什么这样设：

```text
1. lexical coarse routing 当前比粗粒度 Q-K 更稳。
2. Q-K 适合放在 topic 内部做 fine chunk routing。
3. coarse segment gate 是目前最有系统收益的中间层。
4. batch Q-K 和 candidate key cache 是降低在线 routing cost 的工程收益点。
5. neighbor expansion 已验证无收益，不再作为默认。
```

## 5. 最新 neighbor rescue 结论

本地结果：

```text
logs/qmsum_neighbor_rescue_compare_0_30_q5/summary.tsv
```

148 个 query case 上：

```text
neighbor_expand=0:
  selected F1      = 0.1679
  avg turn recall  = 0.1717
  selected KV      = 49.71 MiB
  selected TTFT    = 320.52 ms

neighbor_expand=1:
  selected F1      = 0.1679
  avg turn recall  = 0.1717
  selected KV      = 54.71 MiB
  selected TTFT    = 322.08 ms
```

结论：

```text
neighbor expansion 没有改善质量，只增加 KV 和 TTFT。
```

所以当前默认保持：

```text
route_neighbor_expand=0
```

## 6. 下一步最值得突破的方向

既然 topic/node 可以人工标注，下一步不应该继续纠缠 topic discovery。

更值得推进的是：

```text
人工 topic nodes
-> 更像真实部署的 node packing/layout
-> 更便宜的在线 routing
-> 更清晰的系统指标
```

具体可以拆成三条：

```text
1. 可部署节点构造
   不只是 contiguous/round_robin，而是根据 topic 大小、共现、访问频率、
   transfer segment 合并潜力来打包 topic 到 node。

2. 离线 K-aware 描述符
   每个 chunk/segment 离线存一个轻量 K summary。
   在线先用 Q 和 summary 做便宜筛选，再对少量候选做 exact Q-K。

3. routing-aware transfer layout
   让经常一起被选中的 topic/chunk 更容易被同一个 node 或连续 segment fetch。
```

论文贡献可以这样说：

```text
We assume semantic KV nodes are available from manual or application-level
annotations, and focus on query-aware selective KV transfer under this
deployable node abstraction.
```

不要说成：

```text
We solve automatic meeting topic discovery.
```

## 7. Current Mainline Checkpoint On 2026-06-21

这条主线暂时冻结为当前 baseline，不再补跑 `30:40`，也不继续把
`top2 rescue` / `neighbor expansion` 放进主线。

运行来源：

```text
START_DOC=0 END_DOC=30 MAX_QUERIES=5
--mainline_profile current
CASE_SUMMARY_TAG=current_mainline_smoke
```

注意：这次原本想做 smoke，但命令里 `GPU_ID=0 \ ` 的反斜杠后多了空格，
导致前面的 `START_DOC=0 END_DOC=3 MAX_QUERIES=2` 没有传进脚本，最终使用了
`run_qmsum_current_mainline.sh` 的默认 `0:30, q5`。这个结果可以直接作为
30-doc closeout validation 使用。

结果规模：

```text
docs: 0:30
query cases: 148
```

核心质量指标：

```text
avg full-answer F1:      18.0%
avg selective-answer F1: 16.8%
avg oracle-answer F1:    20.9%
avg selected-full delta: -1.2%
bad output full/sel/oracle: 4.1% / 8.1% / 3.4%
```

核心系统指标：

```text
avg selected KV:         49.7 MiB
avg full KV:             1638.6 MiB
avg KV reduction:        96.2%
avg selected fetch:      19.12 ms
avg full fetch:          554.33 ms
avg fetch reduction:     95.7%
avg selected TTFT:       325.59 ms
avg full TTFT:           569.33 ms
avg TTFT reduction:      29.3%
avg ctx token saving:    94.8%
```

当前瓶颈：

```text
avg online routing:      291.47 ms
avg Q-K model:           271.00 ms
avg exact Q-K:           271.33 ms
avg Q-K total stage:     285.62 ms
avg Q-K candidates:      47.0
```

结论：

```text
current mainline 稳定，质量损失约 1.2 F1 points，通信节省很大。
但是在线 exact Q-K 仍然占 routing/TTFT 的绝大部分。
下一阶段不要继续补小 heuristic，而应该降低 online Q-K 成本。
```

下一阶段方向：

```text
exact online Q-K
-> cheap offline K/chunk descriptor
-> query-time cheap descriptor scoring
-> exact Q-K only as teacher / small-top reranker
```

## 8. CacheGen Compare Baseline

当前新增的 CacheGen 对比是一个外部压缩 baseline，用来回答：

```text
如果不做 selective routing，而是把 full KV 用 CacheGen 压缩后传输，
它的 F1 上限和 TTFT 会是什么量级？
```

入口脚本：

```bash
bash scripts_qmsum/run_qmsum_cachegen_compare.sh
```

需要注意：这一步目前是 `cachegen_full_estimated`，不是“解压后真实生成答案”的评测。

```text
CacheGen-full F1 proxy = full-context answer F1
CacheGen-full TTFT     = measured compressed full-KV bytes + estimated transfer
```

这样做的原因是先建立一个干净的外部基线：

```text
selective routing:
  少传 KV，但要付 online routing / Q-K 成本，并可能损失 answer F1

CacheGen-full:
  不做 selective routing，保留 full-context 质量 proxy，
  但传输的是压缩后的 full KV
```

如果这个 estimated baseline 有价值，下一步再做更严格版本：

```text
CacheGen compress full KV
-> CacheGen decompress
-> convert back to HF past_key_values
-> continue generation
-> compute actual decompressed-KV answer F1
```
