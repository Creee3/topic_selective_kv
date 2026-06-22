# Next Steps Checklist

## Current Status

We are currently working on a **distributed KV routing simulation** rather than a full distributed system.

Current agreed assumption:

```text
QMSum's labeled topics can directly define semantic nodes.
Turn -> topic assignment can also rely on the dataset annotations for now.
```

So the current stage is:

```text
not automatic topic construction
but query routing and fine-grained evidence selection under labeled semantic partitions
```

The current mainline pipeline is:

```text
QMSum meeting
-> build semantic topic nodes from topic_list
-> lexical coarse routing over topic nodes
-> run chunk-level Q-K routing inside chosen topics
-> compare selected topics / turns with relevant_text_span
-> record transfer-unit and transfer-segment statistics
-> compare selective answer quality against full-context answer quality
```

## Current Main Conclusion

From the current experiments:

```text
1. Flat node-level Q-K scoring is too coarse.
2. Chunk-level Q-K retrieval is much better than direct node scoring.
3. For hierarchical routing, pure Q-K topic selection is too weak.
4. Lexical coarse topic routing + Q-K chunk routing is the current best direction.
5. Labeled topic information is currently useful and acceptable in the mainline.
```

Current best hierarchical prototype:

```text
routing_granularity=hierarchical
hier_top_strategy=lexical
hier_top_topics=1
route_chunk_size=128
route_top_k=12
route_per_head=True
```

Updated interpretation of this prototype:

```text
It is a semantic-node routing simulation:
  choose the right labeled topic first,
  then do fine evidence selection inside it.
```

Current immediate optimization direction:

```text
Keep the coarse stage fixed at:
  lexical + top1 topic

Then focus on:
  1. answer-quality validation
  2. fine-stage chunk selection refinement
```

Current result of the previous experiment:

```text
The first hybrid attempt underperformed badly.
So pure embedding remains the current best top-level topic strategy.
The next experiment should compare embedding against rerank-style fusion.
```

Latest update after the cheap coarse-router exploration:

```text
The old statement above is no longer the full picture.

What changed:
  precomputed embedding alone still fails on the canonical hard case,
  but a new lexical-family coarse router fixes that top-1 mistake.

So the next comparison is no longer only:
  embedding vs rerank

It is now:
  embedding
  vs lexical
  vs lexical_prf
  vs lexical_hybrid
  vs rrf
```

## Today

- [x] Reconfirm that the current QMSum line uses labeled topic nodes.
- [x] Reconfirm that the current focus is routing simulation, not auto-topic generation.
- [x] Run the focused top1-topic chunk-budget sweep.
- [x] Compare `route_top_k = 4 / 8 / 12 / 16`.
- [x] Run the topic-label ablation and verify that label-aware coarse routing helps a lot.
- [x] Save the summary output locally.
- [x] Add answer generation into the QMSum evaluation loop.
- [x] Add multiple cheap coarse-routing variants:
  - `lexical`
  - `lexical_prf`
  - `lexical_hybrid`
  - `rrf`
- [x] Run the first coarse-strategy comparison on the canonical hard example.
- [x] Expand the new coarse-strategy comparison from `n=1` to a small multi-sample setting.
- [x] Decide that `lexical` should replace `embedding` as the active top-level default.
- [x] Freeze one clean mainline setting after the multi-sample validation.
- [x] Choose `route_top_k=12` as the current balanced mainline default.

Suggested command:

```bash
bash scripts_qmsum/run_qmsum_mainline_answer_eval.sh
```

## Tomorrow

- [ ] Run the fixed mainline answer evaluation on a small multi-sample setting.
- [ ] Suggested first setting:
  - `START_DOC=5 END_DOC=10 MAX_QUERIES=1`
- [ ] Record:
  - selected-turn F1
  - selective answer F1
  - answer F1 delta
  - context-token saving
- [ ] Run the first fine-stage refinement compare:
  - `route_neighbor_expand=0`
  - `route_neighbor_expand=1`
  - `route_neighbor_expand=2`
- [ ] Decide whether local neighbor expansion helps answer quality enough to justify extra transfer.

Updated short conclusion after the chunk-budget sweep:

```text
Inside the chosen topic, increasing chunk budget still helps.
Under lexical coarse routing, route_top_k=12 is the current best balance point.
```

Updated short conclusion after the current code change:

```text
We are now keeping the two-stage pipeline fixed
and only improving the coarse topic-routing score.
```

Updated short conclusion after the first hybrid experiment:

```text
Naive embedding+Q-K score mixing hurts accuracy.
The next fusion should be selective reranking, not direct weighted averaging.
```

Updated short conclusion after the label ablation:

```text
If semantic topic labels are allowed to define nodes,
the current routing pipeline becomes much more accurate.
So the next bottleneck is no longer "whether nodes have meaning",
but how to preserve answer quality while reducing transferred evidence.
```

Updated short conclusion after the latest coarse-strategy experiment:

```text
The main bottleneck has shifted.
Coarse topic routing is now much stronger under lexical scoring,
so the next real bottleneck is:
  answer preservation
  + fine-stage chunk packing / expansion quality
```

## This Week

- [x] Stabilize one labeled-topic QMSum mainline setting after the cheap coarse-router comparison.
- [ ] Keep answer-generation evaluation as part of every mainline comparison.
- [ ] Compare selective answer quality against full-context answer quality.
- [ ] Read the transfer-accounting outputs together with answer quality.
- [ ] Test at least one fine-stage refinement on top of Q-K chunk selection.
- [ ] Add at least one tensor-statistic fine-stage baseline inspired by the advisor discussion.

## QMSum Plan

Current QMSum structure use:

```text
data/qmsum_structured/train.jsonl
data/qmsum_structured/val.jsonl
data/qmsum_structured/test.jsonl
```

Current hierarchical target:

```text
meeting_transcripts -> full meeting prompt
topic_list -> top-level topic nodes
specific_query_list[i].query -> routing query
specific_query_list[i].relevant_text_span -> ground truth evidence span
```

Current prototype meaning:

```text
topic routing should answer:
  "Which semantic region should we enter?"

chunk routing should answer:
  "Which fine-grained evidence inside that region should we fetch?"

transfer accounting should answer:
  "If we only fetch those chunks, how many transfer units/segments remain?"
```

## Commands

### ShareGPT stability check

```bash
bash _archived_unused_20260606/run_sharegpt_stability_check.sh
```

Note:

```text
This script runs:
0:60
60:120
120:180

chunk_node_score_mode defaults to selected_count
```

Key output files:

```text
logs/sharegpt_stability_check/summary.txt
logs/sharegpt_stability_check/summary.tsv
```

### Parallel ShareGPT stability check

```bash
bash _archived_unused_20260606/run_sharegpt_stability_parallel.sh
```

Default GPUs:

```text
1 2 5
```

Override example:

```bash
GPUS="1 2 7" bash run_sharegpt_stability_parallel.sh
```

### One-click aggregation sweep

```bash
bash scripts_sharegpt/run_chunk_routing_sweep.sh
```

Key output files:

```text
logs/chunk_node_score_sweep/summary.txt
logs/chunk_node_score_sweep/summary.tsv
```

### Hierarchical QMSum sweep

```bash
bash _archived_unused_20260619/scripts_qmsum/run_qmsum_hierarchical_sweep.sh
```

Key output files:

```text
logs/qmsum_hierarchical_sweep/summary.txt
logs/qmsum_hierarchical_sweep/summary.tsv
```

### Focused top1-topic chunk-budget sweep

```bash
bash scripts_qmsum/run_qmsum_top1_chunk_budget_sweep.sh
```

### Cheap coarse-strategy compare

```bash
bash scripts_qmsum/run_qmsum_coarse_strategy_explore.sh
```

Recommended small validation run:

```bash
PARALLEL_RUN=0 \
GPU_LIST="2 3" \
START_DOC=5 END_DOC=8 MAX_QUERIES=1 \
bash scripts_qmsum/run_qmsum_coarse_strategy_explore.sh
```

Recommended next mainline run after the coarse-strategy compare:

```bash
bash scripts_qmsum/run_qmsum_mainline_answer_eval.sh
```

Reason:

```text
We have already chosen the current balanced chunk budget:
  lexical + top1 topic + route_top_k=12

So the next mainline question becomes:
  with this fixed routing setting,
  can we preserve answer quality while keeping transfer low?
```

Deprecated fine-stage refinement:

```bash
bash scripts_qmsum/run_qmsum_neighbor_expand_compare.sh
```

Status:

```text
Do not use this as the next mainline step.

On docs 0:30, max 5 queries/doc, neighbor_expand=1 did not improve selected
F1 or turn recall. It only increased selected KV and selected TTFT.
Keep route_neighbor_expand=0 in the current profile.
```

Key output files:

```text
logs/qmsum_top1_chunk_budget_sweep/summary.txt
logs/qmsum_top1_chunk_budget_sweep/summary.tsv
```

## Current Priority Order

```text
1. Treat docs 0:30 q5 as the temporary current-mainline closeout baseline
2. Stop top2 rescue and neighbor expansion as mainline directions
3. Reframe evaluation as quality + communication + routing cost + TTFT
4. Reduce online exact Q-K cost with cheap offline descriptors
5. Use exact Q-K as teacher or small-candidate reranker, not as the full online router
```

## Not The Priority Right Now

```text
- full distributed multi-GPU system implementation
- too many new datasets at once
- rewriting the whole simulator
- topic-split legacy branch recovery
- automatic topic discovery
- more top2/neighbor rescue sweeps
```
