# Reference Paper And Code Notes

This note summarizes the four main reference papers currently most relevant to
this project:

- CacheGen
- Quest
- KVDirect
- Spotlight

For each one, the goal is to answer four practical questions:

1. What problem does the paper solve?
2. What is the core algorithm or system idea?
3. What part of the local reference code corresponds to that idea?
4. What is most worth borrowing for the current project?

The current project question is:

```text
When long-context KV or text is distributed across many nodes,
can we route a query to only a small number of relevant nodes,
instead of fetching everything?
```

This is not exactly the same as any single reference paper.
The current project is closer to a "cross-paper composition":

```text
Quest-style selective retrieval signal
+ QMSum-style semantic supervision
+ KVDirect-style distributed transfer scenario
+ CacheGen-style compression after selection
```

## Added Classical IR Note: BM25 For Coarse Topic Routing

Although BM25 is not one of the original four system papers above,
it has now become directly relevant to the current QMSum mainline.

Reason:

```text
The current bottleneck moved from fine chunk routing
to coarse topic ranking.

In the hard cases we observed:
  precomputed embedding often keeps the correct topic in top-2,
  but fails to push it to top-1.

A BM25-style lexical scorer is now helping fix exactly that failure mode.
```

### What BM25 is solving in our setting

Classical BM25 solves:

```text
query
-> rank candidate documents by lexical relevance
```

In our current setting, this maps to:

```text
query
-> rank candidate topics by lexical relevance
```

So the translation is:

| Classical IR term | Current QMSum meaning |
|---|---|
| query | user question |
| document | one topic representation |
| collection | candidate topics inside the current meeting |
| BM25 score | lexical topic relevance score |
| retrieval ranking | coarse topic routing order |

### What matters most from the BM25 paper

For our current code, the most important takeaways are not the full probability
derivations, but these four ideas:

1. Ranking is the main goal.

```text
BM25 is a ranking score, not a calibrated probability.
That is enough for coarse topic routing.
```

2. TF saturation matters.

```text
If a token appears many times in one topic,
that should help,
but the gain should saturate rather than grow linearly forever.
```

This is important for our topic representations because words like:

```text
remote
discussion
design
```

may repeat many times inside one topic.

3. IDF matters because nearby topics share many common words.

```text
Common meeting words should not dominate.
Rare discriminative words should matter more.
```

This is exactly why lexical routing helps our nearby-topic hard cases.
If many topics all mention:

```text
remote
design
discussion
```

then a rarer token such as:

```text
presentation
prototype
budget
```

can become the real separator.

4. Length normalization matters because some topic representations are longer.

```text
Longer topic text should not automatically win
just because it has more chances to match query terms.
```

### Why BM25 helps our current bottleneck

Embedding is still useful for semantic similarity,
but in the current QMSum hard cases many neighboring topics already share very
similar semantics.

So the failure mode becomes:

```text
all nearby topics look semantically similar,
but only one of them contains the exact discriminative lexical clues.
```

BM25-style scoring is useful here because it prefers:

- rare query words
- concentrated word occurrence
- topic-specific lexical evidence

rather than only broad semantic similarity.

### What code now corresponds to this idea

Current lexical coarse-routing code:

- `qmsum_routing.py`
  - `tokenize_lexical_text(...)`
  - `build_topic_lexical_document(...)`
  - `build_topic_lexical_stats(...)`
  - `bm25_score_tokens(...)`
  - `score_topics_lexical_qmsum(...)`
  - `score_topics_lexical_prf_qmsum(...)`

Current meaning:

```text
Each topic is treated as a small lexical document.
The query is tokenized.
Then BM25-style lexical ranking is used as a cheap coarse topic router.
```

### What is still approximate in our implementation

Current code is BM25-style, not a full benchmark-faithful BM25 system.

Important simplifications:

- topic representation is built from:
  - topic label
  - representative turns
  - optional speaker summary
- IDF is computed over the current meeting's candidate topics
- lexical scoring is used only for coarse topic ranking
- Q-K chunk routing still handles the fine evidence stage

So the current research interpretation should be:

```text
We are borrowing BM25's ranking principles for topic-level coarse routing,
not claiming a full classical IR reproduction.
```

### What might be worth borrowing next

The BM25 paper also points to one very relevant next improvement for us:

```text
BM25F-style field weighting
```

In our setting this could mean:

- topic label weight
- topic summary weight
- representative turns weight

This is attractive because:

```text
topic labels are short but often highly discriminative,
while representative turns are richer but noisier.
```

So a future improvement could be:

```text
field-aware lexical topic routing
instead of one flat bag of topic tokens
```

## 1. CacheGen

### Paper

- Paper folder:
  `reference_repos/papers/CacheGen/CacheGen.pdf`
- Text extract:
  `reference_repos/papers/CacheGen/CacheGen.txt`
- Code:
  `reference_repos/CacheGen-main`

### What it does

CacheGen is mainly a **KV cache compression and streaming system**.

Its main question is:

```text
If KV cache must be transferred or reused, how can we make it much smaller
without hurting quality too much?
```

### Core idea

At a high level:

- Compress KV cache directly, instead of shortening the original text.
- Use quantization plus arithmetic coding.
- Keep Key more carefully than Value because Key is more sensitive to attention.
- Prepare multiple compressed versions for different bandwidth conditions.

### What the code corresponds to

The current workspace already reuses parts of this stack through:

- `src/utils.py`
- `lmcache/`
- quantization / blob conversion related helpers

The reference implementation itself is not about routing.
It is about:

- KV tensor compression
- storage / loading
- TTFT reduction
- system-level efficiency

### What is worth borrowing now

For the current project, CacheGen is best understood as the **post-routing stage**:

```text
route first
-> choose which remote KV or chunks are worth fetching
-> then compress / transfer only the selected part
```

Most useful ideas to borrow later:

- transmission cost modeling
- adaptive compression after routing
- TTFT-style evaluation

### What it is not

CacheGen is not the main source of the routing algorithm itself.
It helps answer:

```text
After we know what to fetch, how do we fetch it efficiently?
```

## 2. Quest

### Paper

- Paper folder:
  `reference_repos/papers/Quest/Quest.pdf`
- Text extract:
  `reference_repos/papers/Quest/Quest.txt`
- Code:
  `reference_repos/Quest-main`

### What it does

Quest is the most important method reference for the current project.

Its main question is:

```text
During decode, which KV pages are actually worth loading into attention?
```

### Core idea

Quest does not attend to all cached tokens equally.
Instead, it does:

```text
query
-> estimate per-page importance
-> keep top-k pages
-> run sparse attention only on those pages
```

Important details from the paper and code:

- Page-level unit, not full-sequence dense attention
- Query-aware selection
- Explicit top-k budget
- Approximate estimation before exact sparse attention
- Layer sensitivity: early layers may not provide useful sparsity

### What the code corresponds to

Key code path:

- `quest/utils/__init__.py`
  - `decode_estimate(...)`
  - `decode_topk(...)`
  - `decode_sparse_attn(...)`
- `quest/utils/controller.py`
  - page budget handling
- `quest/models/llama.py`
  - `quest_init(...)`

This is a very clean algorithmic pipeline:

```text
estimate
-> top-k select
-> sparse attention
```

### What we already borrowed

The current project has already borrowed Quest's **problem structure**:

```text
Quest:
  choose useful local KV pages

This project:
  choose useful remote nodes / topics / chunks
```

The current project also already borrows:

- top-k budget thinking
- fine-grained selection before coarse transmission
- the idea that query-aware signal should drive selective retrieval

### What we have not borrowed yet

We have not fully borrowed Quest's most elegant implementation ideas yet:

1. Page metadata approximation

Quest uses metadata such as min/max Key summaries.
Current code still relies on heavier direct Q-K style scoring.

2. Explicit layer policy

Quest highlights that some layers are weak for sparsity.
Current project has only partial layer ablation.

3. Strong budget-cost interpretation

Quest's top-k is tightly connected to actual memory movement.
Current project still needs a cleaner communication-cost model.

### Best current takeaway

Quest is the main algorithmic inspiration.

If we later want a stronger and more efficient routing controller, the most
natural next step is:

```text
replace heavy full scoring
with cheaper node/topic/chunk metadata scoring
```

## 3. KVDirect

### Paper

- Paper folder:
  `reference_repos/papers/KVDirect- Distributed Disaggregated LLM Inference.pdf`
- Code:
  `reference_repos/KVDirect-main`

### What it does

KVDirect is mainly about the **distributed transfer layer**.

Its main question is:

```text
If prefill and decode are separated across machines or GPUs,
how do we move KV cache efficiently?
```

### Core idea

The important system idea is:

- distributed / disaggregated inference
- GPU-to-GPU KV transfer
- RDMA / InfiniBand style low-overhead transport
- merging contiguous regions to reduce transfer operations

### What the code corresponds to

The local reference repo contains the system code base and benchmark material,
not a clean small routing algorithm module like Quest.

What matters most conceptually is:

- actual distributed inference scenario
- transfer path after routing
- why "do not fetch everything" matters in practice

### What is worth borrowing now

KVDirect gives the current project its **system-level story**:

```text
Quest tells us how to think about selective retrieval.
KVDirect tells us why distributed transfer makes node selection important.
```

Most useful ideas to borrow later:

- communication cost model
- selected-region coalescing
- routing -> transfer savings narrative

### What it is not

KVDirect is not the main source of the routing score.
It is the main source of the **distributed systems motivation**.

## 4. Spotlight

### Paper

- Paper folder:
  `reference_repos/papers/Spotlight Attention- Towards Efficient LLM.pdf`
- Code:
  `reference_repos/spotlight-main`

### What it does

Spotlight is another selective attention / selective KV paper, but its flavor
is different from Quest.

Its main question is:

```text
Can we aggressively prune KV while preserving accuracy,
using a learned or hash-based approximation?
```

### Core idea

At a high level:

- learn or define a compact proxy space
- compare Q and K in that proxy space
- use the proxy to choose a sparse subset
- then run exact attention on the selected subset

### What the code corresponds to

The repo is broader and more evaluation-heavy.
It is less directly reusable for the current prototype than Quest.

### What is worth borrowing now

Spotlight is valuable mainly as a source of **future baselines and metrics**:

1. Proxy routing idea

Instead of full Q-K or full embedding, use a lighter intermediate signature.

2. Overlap-style evaluation

Measure whether selected units overlap with "truly important" units.

3. Stronger approximate-selection story

This may matter later if the current routing controller becomes too expensive.

### What it is not

Spotlight is probably too heavy as a direct mainline method right now.
It is better treated as:

```text
future stronger baseline / future efficiency variant
```

## Mapping To The Current Project

## Current project stages

The current project can be described as three layers:

### Layer A: Semantic routing structure

Question:

```text
What should a "meaningful node" be?
```

Current answer:

- QMSum topic nodes
- hierarchical topic -> chunk routing

Main supervision source:

- `topic_list`
- `relevant_text_span`

### Layer B: Retrieval / routing signal

Question:

```text
How do we decide which topic/chunk/node is relevant?
```

Current answer:

- embedding for coarse semantic topic routing
- Q-K for fine-grained chunk routing

Main inspiration:

- Quest

### Layer C: Transfer / compression

Question:

```text
After selection, how do we reduce transfer cost?
```

Current answer:

- not fully implemented yet

Main inspiration:

- KVDirect for transfer scenario
- CacheGen for post-selection compression

## What each paper contributes to the current design

| Paper | Most useful role in this project |
|---|---|
| CacheGen | Compression after routing |
| Quest | Main selective-routing algorithm inspiration |
| KVDirect | Distributed transfer scenario and motivation |
| Spotlight | Future approximate-selection baseline |

## Best current interpretation

The current project is no longer just:

```text
Can Q-K score flat nodes?
```

It is becoming:

```text
Can we build a meaningful distributed routing stack:
  semantic coarse routing
  + fine-grained evidence routing
  + eventual transfer/compression savings
```

## What to borrow next

If we prioritize only a few next ideas from the references, the best order is:

1. Quest:
   - stronger budget thinking
   - possible layer ablation
   - metadata-style approximations

2. KVDirect:
   - communication-cost model
   - contiguous region coalescing

3. CacheGen:
   - routing + compression composition

4. Spotlight:
   - cheap proxy-selection baseline
   - overlap-style metric

## Current recommendation

For the immediate next stage, the best practical route is:

```text
keep using:
  embedding topic routing
  + Q-K chunk routing

then add:
  clearer budget sweep
  communication-cost estimation
  flat vs hierarchical comparison
```

This is the most realistic way to connect the current prototype to the reference papers without overcomplicating the code too early.

## Concrete Source Pointers

This section records the most relevant source-level entry points, so later we do
not need to reopen the whole repos again.

### Quest: the cleanest algorithm template

Most useful files:

- `reference_repos/Quest-main/quest/utils/__init__.py`
- `reference_repos/Quest-main/quest/utils/controller.py`
- `reference_repos/Quest-main/quest/models/QuestAttention.py`

Most important flow in code:

```text
decode_estimate
-> decode_topk
-> decode_sparse_attn
```

What each part means:

- `decode_estimate(...)`
  - estimates page importance before exact attention
  - uses page-level metadata instead of full KV
- `decode_topk(...)`
  - applies the explicit page budget
  - turns scores into a fixed-size selected set
- `decode_sparse_attn(...)`
  - runs exact attention only on selected pages
- `InferenceController`
  - manages the page budget, metadata cache, and decode buffers
- `QuestAttention.py`
  - shows the exact runtime branch:
    - if estimate is not needed, do full attention on available pages
    - otherwise estimate, top-k, then sparse attention

Why this matters for us:

```text
Quest is not just "Q-K sparsity".
Its stronger idea is the controller structure:
  cheap estimate
  -> explicit budgeted selection
  -> exact computation on the selected subset
```

That structure is directly reusable for our routing controller:

```text
cheap node/topic estimate
-> top-k node/topic routing
-> exact chunk scoring inside chosen nodes/topics
```

### KVDirect: the transfer path after routing

Most useful file:

- `reference_repos/KVDirect-main/KVDirect/kv_comm.py`

Most important functions:

- `coalesce_indices(...)`
- `KVComm.transfer_kv(...)`

What they mean:

- `coalesce_indices(...)`
  - merges consecutive block indices into contiguous transfer segments
  - reduces the number of send/recv operations
- `transfer_kv(...)`
  - groups requests by remote address
  - iterates layer by layer
  - transfers K and V block regions

Why this matters for us:

```text
Even if routing is correct,
scattered tiny fetches are still expensive.
```

So later our routing evaluation should not only ask:

```text
Did we hit the right topic/turn/chunk?
```

It should also ask:

```text
How fragmented are the selected chunks?
How many transfer segments would KVDirect-style transport need?
Can nearby selected chunks be merged?
```

### Spotlight: proxy-space selection

Most useful file:

- `reference_repos/spotlight-main/spotlight/magicpig/cache_ref-backup.py`

Most important pattern:

```text
Q hash
vs K hash
-> binary-match mask
-> sparse attention over allowed subset
```

What the code is doing:

- projects Q and K into a hash space
- compares hash codes instead of full dot products
- uses the hash agreement mask to keep only a subset
- then performs attention on that filtered subset

Why this matters for us:

```text
Spotlight suggests a future "cheap proxy router":
we do not always need full Q-K or full embedding scoring.
```

This is not the best next mainline step, but it is a good future baseline:

- random
- recency
- embedding
- full Q-K
- cheap proxy hash

### InfiniGen: threshold and budget thinking

Most useful file:

- `reference_repos/InfiniGen-main/speedup/infinigen/infinigen/kv_selection_controller.py`

Most important functions:

- `speculate_attention(...)`
- `select_kv(...)`

What they mean:

- `speculate_attention(...)`
  - uses partial query weight and partial key cache
  - computes a cheap approximate score
  - uses `max - alpha` thresholding
  - then applies a budget-like capped top-k
- `select_kv(...)`
  - gathers the selected KV entries by predicted indices

Why this matters for us:

```text
InfiniGen reminds us that fixed top-k is not the only budget policy.
```

For our routing, later we can compare:

- fixed `route_top_k`
- threshold-based selection
- budget-by-bytes
- budget-by-number-of-chunks

## What These Repos Suggest For Our Next Implementation

If we translate the papers into our current project language, the most natural
next-stage stack is:

```text
Layer 1:
  embedding topic routing
  choose 1 or 2 semantic topic nodes

Layer 2:
  Q-K chunk routing inside chosen topics
  choose top-k evidence chunks

Layer 3:
  merge nearby chosen chunks into fewer transfer segments

Layer 4:
  estimate communication cost

Layer 5:
  later, apply CacheGen-style compression on only the selected chunks
```

This is useful because each reference paper maps to one stage:

| Stage | Reference |
|---|---|
| semantic coarse routing | QMSum supervision + embedding baseline |
| selective fine routing | Quest |
| adaptive threshold / budget variants | InfiniGen |
| transfer coalescing | KVDirect |
| post-selection compression | CacheGen |
| cheap proxy baseline | Spotlight |

## Practical Research Reading Order

If the goal is not "read everything" but "read in the order most useful for our
code right now", the best order is:

1. Quest paper + `quest/utils/__init__.py` + `quest/utils/controller.py`
2. KVDirect `KVDirect/kv_comm.py`
3. InfiniGen `kv_selection_controller.py`
4. Spotlight hash-selection code
5. CacheGen again, but now from the angle of post-routing compression

Why this order:

- Quest explains the cleanest selective-controller structure.
- KVDirect explains why routing quality alone is not enough; transfer pattern matters too.
- InfiniGen helps us think beyond fixed top-k.
- Spotlight is a future cheap approximation baseline.
- CacheGen is most useful after the routing design stabilizes.

## Immediate Discussion Questions After The Current Run Finishes

Once the current cloud run completes, the most useful questions are:

1. Does `hier_top_topics=2` improve turn recall enough to justify broader fetch?
2. Does `route_top_k=8` help more than it hurts precision?
3. Are the chosen chunks concentrated enough that a KVDirect-style coalescing step would save transfers?
4. Is our current bottleneck still top-level topic routing, or has it shifted to in-topic chunk selection?
5. Should the next ablation compare fixed top-k against a threshold-based budget rule?

That is the cleanest way to connect the ongoing experiments with the reference
papers without letting the project drift back into flat, hard-to-interpret
routing.

## Current Fusion Ideas

This section records the most realistic "borrow and fuse" directions after the
latest top1-topic chunk-budget sweep.

Current experimental reading:

```text
topic top-1 hit is still only about 30%
but in-topic chunk budget continues to help up to 16
```

So the next improvement should focus more on **coarse topic routing quality**
than on endlessly increasing chunk budget.

### Fusion idea 1: Quest-style controller at two levels

Best practical interpretation:

```text
level 1:
  cheap topic estimate
  -> top-k topic selection

level 2:
  exact Q-K chunk scoring only inside selected topics
```

What to change in our code later:

- keep current hierarchical structure in `qmsum_sim.py`
- separate "topic score function" from "chunk score function" more clearly
- make the topic selector pluggable:
  - embedding
  - embedding + cheap Q-K hint
  - proxy signature

Why this is the most aligned with Quest:

```text
Quest is really teaching us controller design,
not "always use raw Q-K everywhere".
```

Current status:

```text
This has now moved to a safer rerank direction in qmsum_sim.py:
  embedding first selects candidate topics
  -> Q-K reranks inside that candidate set
```

### Fusion idea 2: KVDirect-style transfer accounting

Current code answers mostly:

```text
Did we find relevant evidence?
```

But the system story needs:

```text
How much transfer would this save?
```

What to add later:

- count selected chunks
- count unique selected topics/nodes
- measure whether selected chunks are contiguous
- estimate "number of transfer segments after coalescing"

This is probably the cleanest next systems-style metric addition.

Current status:

```text
This direction has now been started in qmsum_sim.py.
The current prototype records selected chunk count,
transfer-unit count, transfer-segment count,
and coalescing-style summary metrics.
```

### Fusion idea 3: InfiniGen-style budget policy

Right now we mainly sweep fixed `route_top_k`.

InfiniGen suggests:

```text
score
-> threshold by confidence
-> cap by budget
```

What this could become for us:

- keep chunks whose score is within `max - alpha`
- then cap total selected chunks at a budget ceiling
- compare against fixed `top_k=16`

This is attractive because it may adapt better to:

- easy queries with concentrated evidence
- hard queries with broader evidence

### Fusion idea 4: Spotlight-style cheap proxy baseline

This is not the next mainline change, but it is a strong future ablation:

```text
embedding
vs full Q-K
vs cheap proxy signature
```

What it could look like in our setting:

- build a lightweight topic or chunk signature
- use it for prefiltering before exact Q-K
- then compare quality / cost tradeoff

This is useful if exact Q-K scoring later becomes too expensive at scale.

### Fusion idea 5: CacheGen only after routing stabilizes

Important constraint:

```text
do not mix routing claims and compression claims too early
```

The clean order is still:

```text
prove routing first
-> estimate transfer savings
-> then add CacheGen-style compression on selected chunks
```

Otherwise the story becomes harder to interpret.
