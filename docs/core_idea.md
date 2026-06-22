# Core Idea

## One-Sentence Version

Use the model's own **Q-K attention signal** as a routing signal for distributed KV cache: fetch only the KV chunks/nodes that are relevant to the current query.

## Current Mainline Definition

For the current stage, the idea is intentionally narrower than a full system:

```text
Assume semantic nodes/topics are already given.
Then study whether a query can be routed to the right semantic region,
and whether fine-grained evidence inside that region can be selected efficiently.
```

So the current QMSum line is better summarized as:

```text
query
-> coarse topic routing
-> fine chunk routing
-> evidence-quality evaluation
-> transfer-cost evaluation
-> later answer-quality evaluation
```

## Problem

Long-context LLM inference creates large KV caches. If these KV caches are distributed across nodes, broadcasting or fetching all KV for every new query is wasteful.

The coordinator needs a routing rule:

```text
Given the current query, which remote KV should be fetched?
```

## Proposed Signal

For a query, compute its Q representation. For each candidate KV chunk/node, use the stored K representation to compute a Q-K attention score.

High score means:

```text
The model itself thinks this context is relevant to the current query.
```

This avoids:

- External retrieval indexes.
- Hand-labeled topic boundaries.
- Additional trained routing models.

Important clarification for the current codebase:

```text
The original broad vision was to avoid hand-labeled topic boundaries.
But the current QMSum prototype does not solve that part yet.
At this stage, labeled topics are allowed.
```

That means the active decomposition is:

```text
coarse stage:
  use labeled topic nodes as semantic units

fine stage:
  use Q-K to choose evidence chunks inside the chosen topic
```

## Relationship To Existing Work

| Work | Their question | Our adapted question |
|---|---|---|
| Quest | Which local KV pages should be attended to? | Which distributed KV chunks/nodes should be fetched? |
| InfiniGen | Which offloaded KV tokens should be prefetched? | Which remote KV should be transmitted? |
| CacheGen | How to compress KV for transmission? | Can be used after routing to compress selected KV |
| KVDirect | How to transfer KV efficiently via RDMA? | Can be used as the transport after routing |

## Current Evidence

### Strong Evidence

LongChat chunk selection shows that Q-K based chunk selection can preserve answer quality:

```text
per_head + top_k=2 + chunk_size=256:
  45/50 accuracy
  about 69% token saving
```

The per-head voting design is important because different heads preserve different relevance signals.

### Partial Evidence

Distributed ShareGPT simulation shows that Q-K node scores are:

- Not random.
- Not simple recency.
- Often concentrated on front/mid conversation nodes.

This supports the weaker claim:

```text
Q-K can produce a meaningful routing signal.
```

### Negative/Diagnostic Evidence

Current passkey experiments show that node-level Q-K scoring does **not** reliably locate the node containing an inserted passkey:

```text
Embedding: high passkey hit rate
Q-K node-level scoring: near random passkey hit rate
```

This means the current node-level scoring is too coarse for needle localization.

## Updated Hypothesis

The promising direction is not:

```text
Collapse each node into one scalar score.
```

The promising direction is:

```text
Score smaller units: turn, chunk, or page.
Select global top-k units.
Then map selected units back to nodes for transmission.
```

This makes the method closer to Quest and should better preserve sparse signals like passkeys.

For the current QMSum branch, the hypothesis is even more specific:

```text
The current working two-stage hypothesis is:
  use a cheap coarse topic router to enter the right semantic region,
  then use Q-K for fine evidence selection inside that region.

Recent experiments suggest that lexical coarse routing may currently be stronger
than precomputed embedding on nearby-topic hard cases.
```

## Current Best Interpretation

Recent ShareGPT chunk-routing results suggest:

```text
1. Q-K chunk retrieval itself is already strong.
2. The selected chunk set often covers the answer-bearing node/turn/chunk.
3. The remaining difficulty is the final chunk -> node aggregation step.
4. Counting how many selected chunks fall on each node currently works better than summing or taking a max.
```

Current best observed node aggregation on the ShareGPT passkey setup:

```text
selected_count > selected_max ~= all_chunk_max > selected_sum
```

This supports a more specific claim:

```text
Q-K is useful first as a fine-grained evidence selector, and only second as a node router.
```

## Current Research Boundary

What we are validating now:

```text
1. If topic nodes are already meaningful,
   can we route a query into the correct topic?
2. After entering that topic,
   can Q-K find the relevant chunks/turns?
3. If yes,
   how much transfer can be reduced?
4. Later,
   can the final answer stay close to full-context answering?
```

What we are not focusing on right now:

```text
1. automatic topic discovery
2. full distributed multi-node deployment
3. final transport implementation
4. final online serving optimization
```

## Immediate Next Direction

The next mainline should combine the advisor's suggestion with the current code state:

```text
1. keep labeled topic nodes
2. compare and stabilize the best cheap coarse router
   (currently lexical-style routing looks strongest)
3. keep Q-K fine routing as the main evidence selector
4. add simple tensor-statistic fine-stage baselines:
   mean / variance / max / min / top-k
5. add answer generation and compare:
   full retrieval vs selective retrieval
6. report both answer quality and transfer reduction together
```

## Frozen Mainline On 2026-06-12

The current QMSum mainline is now frozen as:

```text
query
-> lexical coarse topic routing
-> keep top-1 topic
-> Q-K fine chunk routing only inside that topic
-> evaluate evidence coverage
-> evaluate answer quality against full-context answering
```

The main entrypoint is:

```text
qmsum_mainline.py
```

The main routing module is:

```text
qmsum_mainline_routing.py
```

Important implementation details:

- Coarse routing uses BM25-style lexical matching over:
  - pure topic label text
  - plus `topic_prototype_turns=3` representative turns
- The old `"Topic label: ..."` prefix has been removed from the mainline topic representation.
  This keeps the coarse document cleaner and avoids adding useless tokens like
  `topic` and `label`.
- Fine routing uses Q-K attention scores over chunk candidates inside the
  selected topic only.
- Current default fine routing:
  - `route_chunk_size=128`
  - `route_top_k=12`
  - `route_per_head=True`

Research interpretation:

```text
At the current stage,
we are not solving automatic topic discovery.
We assume topic nodes are already given by QMSum topic_list,
then test whether:
  1. coarse routing can enter the correct semantic topic
  2. fine Q-K routing can keep the useful evidence
  3. selective retrieval can save context while keeping answer quality acceptable
```

Recent practical clarifications:

```text
1. Coarse routing now uses:
     pure topic label text
     + 3 prototype turns
2. The old helper prefix "Topic label: ..." has been removed.
3. Fine routing is now understood and implemented as:
     choose topic first
     -> score chunks only inside that topic
4. Mainline logs should expose selected chunk text directly,
   not only selected turn ids.
```

## Current Best QMSum Baseline On 2026-06-16

The strongest validated answer-setting for the current QMSum line is:

```text
query
-> lexical coarse topic routing
-> keep top-1 topic
-> Q-K fine chunk routing only inside that topic
-> dynamic chunk budget
-> qk_then_time answer evidence order
-> selected turns
-> strict answer prompt
```

Observed 25-case answer result:

```text
avg selective-answer F1 = 23.1%
avg answer F1 delta     = +4.2%
selective >= full       = 72.0%
ctx saving              = 86.0%
```

Negative answer-side results that should not be kept as the default:

```text
chunk_turns + strict
answer_aware rerank + turns + strict
```

These results suggest that:

```text
raw selected turns are currently a better answer interface than the tested
repacked alternatives,
as long as the final prompt is kept strict.
```

## Next Research Phase

The current heuristic tuning stage is near a local plateau.

So the next phase should move up one level:

```text
Phase A:
  make QMSum more faithful to distributed selective fetch
  by explicitly grouping topic/chunk KV into virtual nodes before routing

Phase B:
  replace prompt-level heuristics with a two-stage answer interface
  selected turns -> query-oriented notes -> final answer
```

This phase transition matters because the current QMSum code is still:

```text
a routing + transfer simulation,
not yet a physically separated multi-node KV serving system.
```
