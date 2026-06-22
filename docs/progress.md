# Progress

## Cloud Environment

See `docs/cloud_environment.md` for the recorded server/runtime details.

Short version:

- Host: `liuxin@liuxin-1`
- Working dir: `~/working_place/topic_selective_kv`
- Conda env: `cachegen`
- Model: `~/models/mistral-7b/`
- Python `3.10.14`
- PyTorch `2.3.1+cu121`
- Transformers `4.42.3`
- GPUs: `8 x RTX 6000 Ada`

## Completed

### CacheGen Reproduction

Moved to root-level `experiment_log.md`.

Key result:

```text
CacheGen LongChat 50:
  accuracy: 1.00
  compressed KV size: about 172.9 MB
  8-bit baseline size: about 586.7 MB
```

### LongChat Chunk Selection

Script:

```text
batch_eval.py
```

Best current result:

```text
per_head + top_k=2 + chunk_size=256:
  45/50 accuracy
  about 69% token saving
```

Ablation lessons:

- Mean pooling over heads fails badly: `1/50`.
- Per-head voting is the key design.
- `chunk_size=256` is the current sweet spot.
- Single deep-layer scoring can be enough.

### Distributed Routing Simulation

Script:

```text
distributed_sim.py
```

Baseline result:

```text
Q-K node-level routing is not random and not recency.
```

Passkey result:

```text
Embedding finds passkeys well.
Q-K node-level scoring does not.
```

Recent passkey numbers:

| Strategy | top-1 | top-2 |
|---|---:|---:|
| Embedding | about `87.5%` | about `96.4%` |
| Q-K node-level | about `18-21%` | about `49-51%` |
| Random | about `22%` | about `43%` |
| Recency | about `24%` | about `51%` |

Interpretation:

```text
Node-level Q-K aggregation is too coarse for needle/passkey localization.
```

### Chunk-Level Routing Update

Current active setting:

```text
routing_granularity=chunk
route_top_k=4
route_per_head=True
```

Chunk-routing evidence is now clearly stronger than old node-level routing:

```text
selected-node: about 97.9%
selected-turn: about 82.8%
selected-first-chunk: about 79.2%
```

Aggregation sweep result:

| Aggregation mode | top-1 | top-2 | Note |
|---|---:|---:|---|
| `selected_count` | `27.6%` | `52.1%` | Best current final node routing |
| `selected_sum` | `17.7%` | `43.8%` | Worse than count-based |
| `selected_max` | `18.8%` | `47.9%` | Similar to old max-style behavior |
| `all_chunk_max` | `18.8%` | `47.9%` | Old chunk-node ranking logic |

Interpretation:

```text
Chunk retrieval is no longer the main bottleneck.
The main improvement came from changing how selected chunks are aggregated back to nodes.
Count-based aggregation is the best current default.
```

## In Progress

### Hierarchical QMSum Routing

Current design:

```text
meeting transcripts
-> topic nodes from QMSum topic_list
-> top-level semantic routing over topics
-> chunk-level Q-K routing inside selected topics
-> compare selected topics / selected turns against relevant_text_span
-> estimate transfer units / transfer segments / coalescing gain
```

Current scope reminder:

```text
This stage assumes topic structure already exists.
We are not currently solving automatic topic discovery.
QMSum labels are allowed to define:
  1. topic nodes
  2. turn -> topic mapping
```

So the current question is not:

```text
"Can the system discover topics by itself?"
```

The current question is:

```text
"Given meaningful semantic nodes, can we route the query correctly,
 then fetch less KV while still preserving the needed evidence?"
```

Current observations:

```text
Pure Q-K topic routing failed badly.
Embedding topic routing is clearly stronger than Q-K at the coarse semantic level.
The current working hypothesis is:
  semantic coarse routing + Q-K fine routing
```

Current frozen mainline after the latest small-sample validation:

```text
coarse stage:
  lexical topic routing

fine stage:
  Q-K chunk routing inside the selected top1 topic

current balanced default:
  route_top_k=12
```

Refined bottleneck statement:

```text
The main bottleneck is now coarse topic routing, not fine chunk routing.

More specifically:
  precomputed topic embedding often keeps the correct topic in top-2,
  but fails to push it to top-1.

Once hier_top_topics=1 and the top-1 topic is wrong,
the later chunk-routing stage cannot recover.
```

Current accuracy-improvement direction:

```text
Keep the two-stage structure unchanged:
  top-level topic routing
  -> in-topic Q-K chunk routing

But improve the top-level topic routing more safely:
  embedding first selects a few candidate topics
  -> Q-K only reranks inside those candidates
```

Current outcome of the first hybrid attempt:

```text
The first naive hybrid score is worse than pure embedding.
So "direct weighted sum of embedding + Q-K topic score" is not the right fusion yet.
The next active direction is rerank-style fusion instead.
```

Follow-up after the precomputed coarse-routing experiments:

## Frozen Mainline Plan On 2026-06-15

Current confirmed frozen mainline:

```text
lexical coarse topic routing
-> keep top-1 topic
-> Q-K fine chunk routing only inside that topic
-> selected-turn / transfer evaluation
-> full-vs-selective answer F1 comparison
```

Current mainline run already completed on:

```text
20 docs
all queries
route_top_k=12
```

The current saved outputs used as the main reference are:

```text
outputs/qmsum_sim_N4_0_10_mainline_lexical_top1_chunks_12_answers.json
outputs/qmsum_sim_N4_10_20_mainline_lexical_top1_chunks_12_answers.json
outputs/qmsum_answer_log_N4_0_10_mainline_lexical_top1_chunks_12_answers.md
outputs/qmsum_answer_log_N4_10_20_mainline_lexical_top1_chunks_12_answers.md
```

### Mainline Result Snapshot

Current 20-doc result should be remembered as:

```text
total cases: 141
good_or_neutral: 84
negative cases: 57
```

More specifically, among the negative cases:

```text
A_topic_miss: 25
C_topic_hit_turn_hit_answer_drop: 32
```

Interpretation:

```text
The current bottleneck is no longer a single issue.

There are now two main failure sources:
1. coarse topic routing misses the correct topic
2. the topic is correct, but the selected evidence / final answer still degrades
```

### Failure Taxonomy

We now use the following simple taxonomy when reading bad cases:

```text
A_topic_miss
  selected topic is not in relevant topics
  -> later Q-K chunk routing cannot recover

B_topic_hit_turn_miss
  selected topic overlaps relevant topics
  but selected turns still miss the annotated evidence
  -> currently rare in the 20-doc mainline summary

C_topic_hit_turn_hit_answer_drop
  selected topic is correct
  selected turns also hit relevant evidence
  but selected answer F1 is still lower than full-answer F1
  -> this points to chunk precision / chunk completeness / answer-prompt issues
```

### Frozen Next-Step Roadmap

We will follow the next steps in this order and avoid reopening too many side experiments:

```text
Step 1
Stabilize the current mainline reading:
  read the 20-doc outputs carefully
  confirm what already works
  confirm what consistently fails

Step 2
Analyze A_topic_miss cases first:
  why lexical top-1 picked the wrong topic
  what kinds of queries are most confusing
  whether nearby topics or speaker-specific questions are the main cause

Step 3
Analyze C_topic_hit_turn_hit_answer_drop cases next:
  topic is right but answer still gets worse
  inspect whether the issue comes from:
    chunk precision
    chunk completeness
    too much noisy evidence
    answer prompt / generation format

Step 4
Run only a small number of high-value improvements:
  chunk budget
  neighbor expansion
  chunk aggregation / chunk filtering
  answer-prompt adjustments

Step 5
After the mainline is stable enough,
add a distributed-systems comparison note:
  full pull
  selective pull
  remote query/decode
compare transfer cost and answer quality together
```

### Immediate Working Rule

Until the mainline is clearly improved, we should avoid:

```text
1. opening many new coarse strategies again
2. mixing old archived branches into the active reading path
3. judging the system only by one good or one bad case
```

Instead, the immediate workflow should be:

```text
case grouping
-> failure categorization
-> representative-case reading
-> small targeted fix
-> rerun the same mainline evaluation
```

## External Ideas Collected On 2026-06-15

Recent literature/system scan suggests that the most useful additions are not
"replace the whole pipeline", but a small number of targeted ideas that match
the current failure taxonomy.

### Idea Group 1: Better Coarse Topic Reranking

Representative families:

```text
ColBERT / ColBERTv2
COIL
```

Why this is relevant:

```text
The current A_topic_miss failures are usually not "completely unrelated topic"
errors.
They are often near-topic confusions.

So a better next step is:
  cheap lexical top-k first
  -> stronger token-level rerank only inside that tiny candidate set
```

Current research fit:

```text
This matches our current two-stage philosophy well:
  cheap coarse filter
  -> stronger but still limited rerank
```

Recommended adaptation for our codebase:

```text
lexical top-2 or top-3 topics
-> token-level late-interaction rerank
-> keep final top-1 topic
```

Priority:

```text
high
```

### Idea Group 2: Stronger Sparse / Lexical Topic Scoring

Representative families:

```text
DeepCT
SPLADE
```

Why this is relevant:

```text
The current coarse stage is still lexical.
Many failures appear when:
  the query wording and topic wording are close in meaning,
  but not close enough in exact lexical overlap.
```

Current research fit:

```text
This keeps the same explainable coarse-routing story,
but makes lexical weighting smarter.
```

Recommended adaptation for our codebase:

```text
keep BM25 as baseline
add one lightweight stronger sparse variant
compare especially on A_topic_miss cases
```

Priority:

```text
medium
```

### Idea Group 3: Topic Expansion / Pseudo-Query Expansion

Representative family:

```text
doc2query-style expansion
```

Why this is relevant:

```text
Some topic misses are likely caused by wording mismatch:
  topic label says one thing
  query asks the same thing in a different way
```

Current research fit:

```text
This is easy to explain and still keeps the current lexical mainline intact.
It can be presented as enriching the topic document rather than replacing the
router.
```

Recommended adaptation for our codebase:

```text
for each topic:
  add a few pseudo-queries or expanded lexical cues
then rerun lexical routing
```

Priority:

```text
medium-high
```

### Idea Group 4: Two-Stage Fine Chunk Filtering

Representative inspiration:

```text
Quest-style query-aware page/chunk filtering
```

Why this is relevant:

```text
Even after the current cleanup,
fine routing may still carry too many noisy chunks.

The current fine stage may benefit from:
  cheap in-topic prefilter
  -> more precise Q-K reranking
```

Current research fit:

```text
This directly strengthens the current Q-K chunk-routing line
without changing the mainline definition.
```

Recommended adaptation for our codebase:

```text
inside the selected topic:
  prefilter chunk candidates cheaply
  -> then run Q-K on the reduced chunk set
```

Priority:

```text
medium
```

### Idea Group 5: Diversity / De-duplication Before Answering

Representative inspiration:

```text
MMR-style diversity filtering
```

Why this is relevant:

```text
Many C_topic_hit_turn_hit_answer_drop cases suggest:
  the topic is correct
  relevant turns are touched
  but the final selected chunk set is still noisy, repetitive, or incomplete
```

Current research fit:

```text
This is a natural answer-quality fix:
  not only ask "is this chunk relevant?"
  but also ask "is this chunk adding new useful information?"
```

Recommended adaptation for our codebase:

```text
take top-N Q-K chunks
-> apply a diversity / redundancy penalty
-> keep the final answer context
```

Priority:

```text
high
```

### Idea Group 6: Distributed-System Interpretation

Representative system direction:

```text
prefill/decode disaggregation
state transfer / KV transfer
```

Why this is relevant:

```text
The current codebase is closer to:
  selective KV fetch / selective state pull
than to:
  ship the query to a remote node and let it do the whole decode
```

Current research fit:

```text
This helps explain why the current project still has system meaning,
even though it is a routing simulation rather than a full deployment.
```

Recommended adaptation for our writeup:

```text
compare:
  full pull
  selective pull
  remote query/decode
using both transfer cost and answer quality
```

Priority:

```text
high for discussion
lower for immediate code changes
```

## Current Recommended Improvement Order

After combining the current mainline results with the external idea scan,
the recommended implementation order is:

```text
1. analyze A_topic_miss cases carefully
2. try a stronger coarse rerank on top of lexical top-k
3. analyze C_topic_hit_turn_hit_answer_drop cases carefully
4. add diversity / de-duplication on top of selected chunks
5. only then revisit chunk-budget / neighbor-expansion refinements
6. finally summarize full pull vs selective pull vs remote decode
```

## Immediate Next Move

The next concrete action should be:

```text
Take the 25 A_topic_miss cases
-> group them by error pattern
-> understand why lexical top-1 failed
-> then decide whether:
   topic expansion
   lexical top-k rerank
   or a stronger sparse variant
should be implemented first
```

## 2026-06-15 Coarse-Rerank Follow-up

We ran a small controlled check on:

```text
docs 5:8
first 3 queries per doc
9 cases total
```

### Result 1: prototype lexical -> full-topic lexical rerank is useful

Observed pattern:

```text
The rerank-only version repaired some earlier coarse topic misses.
In particular, some previously wrong top-1 topics were corrected,
and selected-turn hit changed from 0 to 1 on representative bad cases.
```

Current interpretation:

```text
Keep coarse lexical rerank.
This is the current best coarse-stage refinement we have validated.
```

### Result 2: lightweight query-style topic expansion is not worth keeping

Tried idea:

```text
Add query-like topic text such as:
  discussion about ...
  summary of ...
  decision about ...
and speaker-style phrases
inside the coarse lexical rerank stage
```

Observed outcome:

```text
Overall topic hit did not improve.
Average turn F1 became slightly worse.
Average answer F1 also dropped relative to rerank-only.
Some individual cases changed to a different "relevant" topic,
but the final evidence and answer quality still got worse.
```

Current interpretation:

```text
The lightweight query-style topic expansion adds more noise than benefit
in the current mainline setting.

So:
  keep rerank-only
  remove the query-style expansion variant
  do not continue this branch as the active mainline
```

### Updated Mainline Decision

The current coarse-stage choice should now be treated as:

```text
prototype lexical coarse scoring
-> top-k candidate topics
-> full-topic lexical rerank
-> final top-1 topic
```

The next mainline improvement should no longer be:

```text
more generic topic expansion text
```

The next mainline improvement should instead move to:

```text
fine-stage chunk filtering / de-duplication / diversity control
```

```text
Several attempts were tested to rescue precomputed coarse routing:
  1. topic_prototype_turns sweep
  2. topic representation template change
  3. candidate-topic rerank

Observed pattern:
  the correct topic is often already in top-2,
  but still does not become top-1.

Interpretation:
  the problem is not simply "too few prototype turns";
  it is poor discrimination between nearby semantically similar topics.
```

Latest small-result snapshot:

```text
hier_top_topics=1
hier_top_strategy=embedding
route_top_k=16
route_per_head=True

selected-topic hit: about 30%
selected-turn hit: about 30%
avg turn recall: about 16.5%
avg turn precision: about 13.6%
avg turn F1: about 11.9%
```

Interpretation:

```text
This is not a strong result yet.
But it is much better than the earlier Q-K-only hierarchical attempt, which was near 0%.
The top-level semantic routing idea appears directionally correct.
Small sweep evidence suggests:
  keep top-level routing tight at 1 topic
  then increase chunk budget inside that topic.
Within the current n=10 slice, larger chunk budget still improves recall/F1 without hurting precision much.
The current bottleneck is increasingly the top-level topic routing stage, not just in-topic chunk budget.
```

Important update after the label-aware topic routing results:

```text
Once labeled topic information is allowed to help the coarse routing stage,
topic hit and turn-level evidence metrics improve a lot.
This means the current QMSum line is now a cleaner simulation of:
  semantic-node routing
  + in-node evidence selection
instead of a topic-discovery experiment.
```

Focused chunk-budget sweep:

| Setting | selected-topic hit | selected-turn hit | recall | precision | F1 |
|---|---:|---:|---:|---:|---:|
| `top1_chunks_4` | `30.0%` | `30.0%` | `6.5%` | `12.9%` | `6.6%` |
| `top1_chunks_8` | `30.0%` | `30.0%` | `11.8%` | `13.6%` | `10.4%` |
| `top1_chunks_12` | `30.0%` | `30.0%` | `12.9%` | `13.5%` | `11.0%` |
| `top1_chunks_16` | `30.0%` | `30.0%` | `16.5%` | `13.6%` | `11.9%` |

Most important reading of this table:

```text
1. More chunk budget is still helping inside the selected topic.
2. Precision stays roughly stable while recall rises.
3. selected-topic hit remains stuck at 30%.
4. Therefore the current main bottleneck is coarse topic routing accuracy.
```

Embedding vs first hybrid topic-strategy comparison:

| Strategy | top-1 | top-2 | selected-topic hit | selected-turn hit | recall | precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `embedding` | `30.0%` | `40.0%` | `30.0%` | `30.0%` | `16.5%` | `13.6%` | `11.9%` |
| `hybrid` | `10.0%` | `20.0%` | `10.0%` | `10.0%` | `8.2%` | `8.0%` | `8.1%` |

Interpretation of this comparison:

```text
1. The current hybrid formula hurts topic routing accuracy.
2. That accuracy drop propagates to turn-level evidence quality.
3. The lower transfer cost of hybrid is not useful if accuracy collapses.
4. So we should keep pure embedding as the current top-level default.
5. If we fuse Q-K later, it should be as a reranker or filter, not as a naive weighted sum.
```

Label ablation snapshot now worth remembering:

| Case | top-1 | top-2 | selected-topic hit | selected-turn hit | recall | precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `topic_label_weight=0.0` | `50.0%` | `83.3%` | `50.0%` | `50.0%` | `39.8%` | `35.5%` | `35.0%` |
| `topic_label_weight=0.35` | `86.7%` | `90.0%` | `86.7%` | `86.7%` | `66.0%` | `64.3%` | `60.2%` |

Interpretation:

```text
At the current stage, semantic labels are not "cheating";
they define the virtual node meaning.
Under that assumption, the routing pipeline becomes much stronger and much closer to the advisor's intended setting.
```

Current limitation:

```text
Although the logic is hierarchical,
the implementation is still a research prototype rather than a fully optimized system path.
Some chunk Q-K scoring is still computed more broadly than a final production router would allow.
So current conclusions are mainly about routing quality, not final system efficiency.
```

Important answer-evaluation update:

```text
The current QMSum line now also evaluates:
  full-context answer
  vs
  selective-context answer

using token-level F1 against the dataset gold answer.

So the end-to-end evaluation stack is now:
  coarse topic hit
  -> selected-turn evidence quality
  -> transfer reduction
  -> final answer quality
```

Latest coarse-routing breakthrough:

```text
To go beyond precomputed embedding alone, a new cheap coarse-routing family was added:
  lexical
  lexical_prf
  lexical_hybrid
  rrf

The motivation is:
  nearby topics in QMSum are often semantically close,
  so pure embedding can blur them together,
  while lexical cues may separate them better.
```

Canonical hard-case result worth remembering:

```text
Case:
  doc=5, query=0
  query="Summarize the presentation and discussion about the design of the remote."

Previous precomputed-embedding result:
  top-1 topic wrong
  selected-topic hit = 0%
  selected-turn hit = 0%
  selective answer F1 = 20.7%

New lexical-family result:
  lexical / lexical_prf / lexical_hybrid / rrf all pick the correct top-1 topic
  selected-topic hit = 100%
  selected-turn hit = 100%
  selected-turn F1 = 85.2%
  selective answer F1 = 25.7%
```

Interpretation of this new result:

```text
This is strong evidence that the current failure mode really is coarse topic routing.
For the canonical hard example, lexical-style coarse routing fixes the exact
top-1 topic mistake that precomputed embedding could not fix.

However, this is still only n=1.
So it is a strong directional signal, not yet a stable final conclusion.
```

Current immediate validation plan:

```text
1. Keep lexical as the frozen coarse mainline.
2. Validate answer quality on the fixed `route_top_k=12` setting.
3. Compare fine-stage variants without changing the coarse router.
4. Read answer quality together with transfer reduction.
```

Recommended next command:

```bash
PARALLEL_RUN=0 \
GPU_LIST="2 3" \
START_DOC=5 END_DOC=8 MAX_QUERIES=1 \
bash scripts_qmsum/run_qmsum_coarse_strategy_explore.sh
```

Current knobs:

- `hier_top_topics`
- `hier_top_strategy`
- `route_top_k`
- `route_chunk_size`
- `route_per_head`

## To Do

1. Keep labeled topic nodes as the active assumption for the QMSum mainline.
2. Keep `lexical` as the current default coarse router.
3. Keep Q-K as the current default fine-grained selector inside the chosen topic.
4. Treat `route_top_k=12` as the current balanced mainline default.
5. Validate answer quality under this fixed mainline.
6. Add fine-stage refinements on top of the fixed mainline:
   - local neighbor chunk expansion
   - later dynamic budget
   - later statistic-based rerank
7. Add simple fine-stage statistical baselines inspired by the advisor discussion:
   - mean
   - variance
   - max / min
   - top-k values
8. Compare those fine-stage baselines against the current Q-K chunk selector.
9. Continue recording KVDirect-style communication accounting:
   - selected chunks
   - transfer units
   - transfer segments
   - coalescing gain
10. Keep claims separated very clearly:
   - labeled topics define semantic nodes
   - the coarse router is still under active comparison
   - Q-K is currently best used as fine evidence routing
   - routing should reduce transfer
   - answer quality is now in the loop and should stay in the loop

## 2026-06-12 Mainline Freeze

The current active QMSum mainline is now:

```text
lexical coarse topic routing
-> top-1 topic only
-> Q-K fine chunk routing inside the selected topic
-> evidence evaluation
-> answer evaluation against full-context answering
```

Main code path:

```text
qmsum_mainline.py
qmsum_mainline_routing.py
qmsum_data.py
qmsum_eval.py
qmsum_answering.py
qmsum_output.py
```

Mainline cleanup that has now been applied:

```text
The coarse topic document no longer prepends:
  "Topic label: "

The topic label is now used as pure label text.
So the mainline no longer injects useless lexical tokens like:
  topic
  label
```

Current default knobs:

- `hier_top_topics=1`
- `hier_top_strategy=lexical`
- `route_chunk_size=128`
- `route_top_k=12`
- `route_per_head=True`
- `route_neighbor_expand=0`

## 20-Doc Mainline Answer Snapshot

Command:

```bash
GPU_A=2 GPU_B=3 MAX_QUERIES=0 ROUTE_TOP_K=12 \
bash _archived_unused_20260619/scripts_qmsum/run_qmsum_mainline_answer_eval_20docs_dual_gpu.sh
```

Observed shard `docs 0:10`:

```text
n=65
selected-topic hit: 72.3%
selected-turn hit: 72.3%
avg turn recall: 47.4%
avg turn precision: 38.6%
avg turn F1: 36.9%
avg full-answer F1: 20.2%
avg selective-answer F1: 20.6%
avg F1 delta: +0.4%
avg context token saving: 87.0%
selective >= full: 58.5%
```

Observed shard `docs 10:20`:

```text
n=76
selected-topic hit: 77.6%
selected-turn hit: 77.6%
avg turn recall: 52.7%
avg turn precision: 42.0%
avg turn F1: 40.3%
avg full-answer F1: 20.2%
avg selective-answer F1: 21.5%
avg F1 delta: +1.2%
avg context token saving: 86.8%
selective >= full: 60.5%
```

Current interpretation:

```text
1. The current lexical coarse router is usable but still imperfect.
2. Once the correct topic is entered, the fine Q-K stage usually keeps a useful
   portion of the relevant evidence.
3. Selective retrieval is already reducing context by about 87%.
4. On the current 20-doc snapshot, selective answering is roughly comparable to
   full-context answering on average, with a small positive F1 delta.
5. The main bottleneck remains coarse top-1 topic accuracy.
```

## 2026-06-15 Clarifications

Recent clarifications that should now be treated as part of the mainline understanding:

```text
1. Coarse routing and fine routing are playing different roles.
2. Coarse routing is a lexical/BM25-style topic selector.
3. Fine routing is a Q-K chunk selector inside the chosen topic only.
4. The answer log should be readable enough to inspect selected evidence directly.
```

### Coarse Routing Clarification

The current coarse topic document is:

```text
pure topic label text
+ 3 prototype turns
```

More concretely:

- topic labels come from QMSum `topic_list`
- prototype turns are sampled by position from the turns covered by that topic
- current default is `topic_prototype_turns=3`
- stopword filtering is applied before BM25-style scoring
- this is still the same lexical routing idea, not a different learned model

Important cleanup:

```text
The old helper prefix "Topic label: ..." has been removed.
So coarse routing no longer injects the useless tokens:
  topic
  label
```

### Fine Routing Clarification

The current fine stage does:

```text
selected top-1 topic
-> build chunk candidates only from that topic
-> compute Q-K chunk scores
-> keep selected chunks
```

Important reminder:

```text
The mainline is now coarse-first in implementation as well:
it does not score chunks from every topic and then filter later.
It first picks the topic, then scores only chunks inside that topic.
```

### Output Logging Clarification

The answer markdown / jsonl outputs are now expected to expose selected chunk text
more directly.

That means each case log should be read as:

```text
query
-> selected topic
-> selected turns
-> selected chunks
-> chunk text
-> gold/full/selected answers
```

This change is important because it makes it much easier to explain:

```text
which chunk was selected,
what text it actually contains,
and whether the selected evidence is semantically reasonable.
```

## Reserved Future Datasets On 2026-06-15

These datasets are now downloaded and reserved for possible later use,
but they are not part of the active QMSum mainline yet.

Current local/cloud paths:

```text
data/hotpotqa/
  train.jsonl
  validation.jsonl
  test.jsonl

data/kilt/
  train.jsonl
  validation.jsonl
  test.jsonl
```

Planned future role:

```text
HotpotQA:
  likely candidate for multi-node evidence routing over Wikipedia-style documents
  useful when we want query -> article/paragraph/chunk style routing

KILT:
  likely candidate as a more unified knowledge-intensive benchmark base
  useful when we want a cleaner "knowledge node" interpretation
```

Important boundary:

```text
These two datasets are only reserved for future exploration.
They should not distract from the current active mainline:
  QMSum lexical coarse topic routing
  -> top-k candidate rerank
  -> top-1 topic
  -> in-topic Q-K chunk routing
  -> answer/transfer evaluation
```

## 2026-06-15 Final Coarse-Fine Optimization Round

We searched for one last coarse-fine improvement direction before moving away
from this layer. The useful outside ideas are:

```text
Self-RAG:
  do not retrieve a fixed amount for every query; retrieval should be adaptive.

Quest:
  KV/page selection should be query-aware, not a static cache cut.

Lost in the Middle:
  answer quality depends not only on whether evidence is selected,
  but also where selected evidence is placed in the final context.

LLMLingua / LongLLMLingua:
  compression should preserve key information density under a budget,
  rather than only minimizing token count.
```

References:

```text
Self-RAG:
  https://arxiv.org/abs/2310.11511

Quest:
  https://arxiv.org/abs/2406.10774

Lost in the Middle:
  https://arxiv.org/abs/2307.03172

LongLLMLingua:
  https://arxiv.org/abs/2310.06839
```

Current decision:

```text
The default mainline returns to rerank-only:
  lexical prototype topic routing
  -> full-topic lexical rerank over top candidates
  -> top-1 topic
  -> in-topic Q-K chunk selection

Hard and soft diversity are kept as optional compression diagnostics.
They are not the default because they reduced evidence recall on the 3-doc
small check.
```

New final experiment added:

```text
query-type dynamic budget + evidence ordering
```

Implementation meaning:

```text
1. Classify the query with a simple heuristic:
   summary-like / detail-like / balanced.
2. Use different fine-stage chunk budgets:
   summary-like  -> more chunks, because summary questions need broader evidence.
   detail-like   -> fewer chunks, because decision/fact questions should be tighter.
   balanced      -> keep the current route_top_k=12 behavior.
3. Optionally order selected answer evidence by:
   time
   qk
   qk_then_time
```

Why this is the right last coarse-fine attempt:

```text
Our 20-doc run shows topic routing is usable but not perfect.
The larger issue now is answer stability: some selected contexts are too sparse
or poorly organized even when the topic is correct.
Dynamic budget and evidence ordering directly target that failure mode without
opening another broad coarse-router search.
```

Small validation protocol remains:

```bash
python qmsum_mainline.py \
  --model_id ~/models/mistral-7b/ \
  --start_doc 5 \
  --end_doc 8 \
  --max_queries_per_doc 3 \
  --eval_answers \
  --route_top_k 12 \
  --route_per_head \
  --dynamic_route_budget \
  --dynamic_summary_top_k 16 \
  --dynamic_detail_top_k 8 \
  --dynamic_balanced_top_k 12 \
  --answer_evidence_order qk_then_time \
  --case_summary_tag dynamic_budget_smallcheck
```

Compare against:

```text
rerank_smallcheck:
  selected answer F1 20.4%, F1 delta +2.4%, ctx saving 85.6%

rerank_diverse_smallcheck:
  selected answer F1 20.5%, F1 delta +2.5%, ctx saving 96.8%
  but turn recall dropped to 26.4%

rerank_diverse_soft_smallcheck:
  selected answer F1 19.2%, F1 delta +1.3%, ctx saving 90.3%
  turn recall 45.2%
```

### Dynamic Budget Small Check Result

The corrected dynamic-budget run on docs 5:8, first 3 queries per doc showed:

```text
n=9
avg selected answer F1: 20.57%
avg full answer F1:     17.94%
avg F1 delta:           +2.64%
ctx token saving:       85.75%
selective >= full:      55.6%
```

Interpretation:

```text
Dynamic budget + qk_then_time ordering is safe and gives a small improvement
over the rerank-only small check, but the gain is modest.
```

Follow-up fix:

```text
The first dynamic-budget heuristic classified:
  "What is the decision of the discussion ..."
as summary because it saw "discussion".

We changed the rule so detail-like intents are checked first.
This means decision / what did / what could / think / how many style queries
now win over broad summary words like discussion.
```

Next validation scale:

```text
Run docs 5:10 with max_queries_per_doc=5.
This gives up to 25 cases and is the next check before deciding whether to keep
dynamic budget as the default mainline option.
```

### Dynamic Budget 25-Case Result

Command setting:

```text
docs 5:10
max_queries_per_doc=5
route_top_k=12
dynamic budget enabled
answer_evidence_order=qk_then_time
```

Observed result:

```text
n=25
lexical top-1 topic hit:      88.0%
selected-turn hit:           88.0%
avg turn recall:             56.7%
avg turn precision:          56.1%
avg turn F1:                 51.5%
avg full-answer F1:          20.0%
avg selective-answer F1:     22.1%
avg answer F1 delta:         +2.1%
avg context token saving:    86.3%
selective >= full:           52.0%
```

Current decision:

```text
Dynamic budget is safe and mildly positive, but the gain is not large enough
to keep searching only at the coarse/fine routing level.

The main bottleneck has moved to answer generation:
  selected evidence can hit the right topic/turns
  but the final selected answer can still be repetitive, incomplete, or poorly
  organized.
```

### Answer Optimization Mainline

New implementation direction:

```text
Keep the current routing mainline fixed:
  lexical coarse routing
  -> top-1 topic
  -> in-topic Q-K chunk selection
  -> dynamic budget / evidence ordering

Then optimize only the selected-answer prompt/context:
  turns       = old behavior, answer from selected turns
  chunk_turns = answer from full turns corresponding to selected chunks
  chunks      = answer from selected chunk text directly

Also add a stricter prompt style to reduce:
  repeated Question/Answer loops
  copied irrelevant filler
  uncontrolled continuation
```

Code options:

```text
--selected_answer_context_mode {turns,chunk_turns,chunks}
--answer_prompt_style {basic,strict}
--answer_evidence_max_entries
--answer_evidence_max_chars
```

Next small validation command:

```bash
GPU_ID=2 \
START_DOC=5 \
END_DOC=10 \
MAX_QUERIES=5 \
ROUTE_TOP_K=12 \
DYNAMIC_ROUTE_BUDGET=1 \
DYNAMIC_SUMMARY_TOP_K=16 \
DYNAMIC_DETAIL_TOP_K=8 \
DYNAMIC_BALANCED_TOP_K=12 \
ANSWER_EVIDENCE_ORDER=qk_then_time \
SELECTED_ANSWER_CONTEXT_MODE=chunk_turns \
ANSWER_PROMPT_STYLE=strict \
ANSWER_EVIDENCE_MAX_ENTRIES=80 \
ANSWER_EVIDENCE_MAX_CHARS=600 \
CASE_SUMMARY_TAG=answer_strict_chunkturns_5docs_q5 \
LOG_DIR=logs/qmsum_answer_strict_chunkturns_5docs_q5 \
bash scripts_qmsum/run_qmsum_mainline_answer_eval.sh
```

How to judge:

```text
Compare against dynamic_budget_detail_first_5docs_q5:
  selected-answer F1: 22.1%
  selective >= full: 52.0%
  context saving:    86.3%

Keep the answer optimization only if it improves answer F1 and/or
selective >= full without destroying context saving.
```

### Answer Optimization First Result

Tested setting:

```text
selected_answer_context_mode = chunk_turns
answer_prompt_style          = strict
dynamic budget               = on
answer_evidence_order        = qk_then_time
```

Observed result on docs 5:10, max_queries_per_doc=5:

```text
n=25
avg full-answer F1:          19.0%
avg selective-answer F1:     19.2%
avg answer F1 delta:         +0.3%
avg context token saving:    87.1%
selective >= full:           48.0%
```

Interpretation:

```text
This answer-interface variant did not improve the mainline.
Routing metrics stayed unchanged, so the drop is attributed to answer-side
formatting rather than evidence selection quality.
```

Next ablation:

```text
We should separate:
  A. prompt-style effect      -> turns + strict
  B. evidence-format effect   -> chunk_turns + basic

If turns + strict also drops, the strict prompt is the problem.
If chunk_turns + basic also drops, the evidence format is the problem.
If only one drops, we isolate the culprit directly.
```

Dual-GPU helper script:

```bash
bash scripts_qmsum/run_qmsum_answer_ablation_dual_gpu.sh
```

### Answer Improvement Literature Notes

Priority group A: closest to the current bottleneck and easiest to connect to
our mainline.

1. Lost in the Middle: How Language Models Use Long Contexts
   paper:
   [arXiv 2307.03172](https://arxiv.org/abs/2307.03172)

   key idea:
   relevant evidence placement matters; models often use information near the
   beginning or end of context better than information buried in the middle.

   what it suggests for us:
   answer quality may improve even without changing routing if we repack
   selected evidence more carefully.

   directly usable idea:
   test evidence packing strategies such as:
   - strongest 2 evidence items first
   - strongest 2 evidence items last
   - conclusion-like evidence first, supporting detail later

2. LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression
   paper:
   [arXiv 2310.06839](https://arxiv.org/abs/2310.06839)

   key idea:
   prompt compression can improve both efficiency and answer quality when it
   preserves dense key information instead of passing long noisy context.

   what it suggests for us:
   the current hard truncation by max chars is probably too crude; we should
   try content-aware evidence compression instead of simple clipping.

   directly usable idea:
   compress each selected evidence item into one short query-relevant note
   before final answer generation.

3. RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation
   paper:
   [arXiv 2310.04408](https://arxiv.org/abs/2310.04408)

   key idea:
   retrieved documents do not have to be prepended in raw form; they can first
   be compressed extractively or abstractively into shorter answer-useful text.

   what it suggests for us:
   our selected chunks may already be correct, but raw turn/chunk text may be a
   poor interface for answer generation.

   directly usable idea:
   selected chunks
   -> compress into short extracted facts or notes
   -> then generate the final answer

4. Learning to Rank Utterances for Query-Focused Meeting Summarization
   paper:
   [arXiv 2305.12753](https://arxiv.org/abs/2305.12753)

   key idea:
   QMSum-style tasks benefit from ranking utterances for the query first, then
   using top utterances for generation.

   what it suggests for us:
   retrieval quality for answer generation is not just "is this relevant", but
   also "is this among the best answer-supporting utterances".

   directly usable idea:
   add a light answer-aware rerank on top of selected chunk/turn evidence using
   lexical overlap, Q-K score, and answer-cue bonuses.

5. Improving Query-Focused Meeting Summarization with Query-Relevant Knowledge
   paper:
   [arXiv 2309.02105](https://arxiv.org/abs/2309.02105)

   key idea:
   the long-input difficulty in QMSum comes from sparse query-relevant signals,
   and an explicit knowledge-aware two-stage pipeline helps both extraction and
   final generation.

   what it suggests for us:
   evidence should be rewritten into query-oriented knowledge units before
   answering, rather than sent as raw conversation fragments.

   directly usable idea:
   build note-style evidence entries such as:
   speaker | main fact | relation to query

6. DYLE: Dynamic Latent Extraction for Abstractive Long-Input Summarization
   paper:
   [arXiv 2110.08168](https://arxiv.org/abs/2110.08168)

   key idea:
   long summarization improves when extraction and generation are connected more
   tightly than a simple hard pipeline.

   what it suggests for us:
   our current hard handoff from selected evidence to answer generation may be
   too brittle; a softer notion of evidence importance may help later.

   directly usable idea:
   in the short term, use weighted evidence ordering or weighted note packing
   rather than only binary keep/drop decisions.

Priority group B: useful inspiration, but currently heavier than what we want
to implement first.

7. Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models
   paper:
   [arXiv 2311.09210](https://arxiv.org/abs/2311.09210)

   key idea:
   instead of answering directly from retrieved documents, first generate
   reading notes, then answer from those notes.

   why it matters:
   this is the closest literature match to the note-style answer interface we
   are now considering.

   caution:
   the full method is heavier than our current prototype, so we should only
   borrow the note-construction idea first.

8. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
   paper:
   [arXiv 2310.11511](https://arxiv.org/abs/2310.11511)

   key idea:
   retrieval and generation quality can improve if the model can critique its
   own evidence use and answer faithfulness.

   why it matters:
   it suggests a later answer-side verifier or post-check stage.

   caution:
   this line usually implies extra training or more complex inference control,
   so it is not the first next step for this repo.

9. RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation
   paper:
   [arXiv 2408.08067](https://arxiv.org/abs/2408.08067)

   key idea:
   retrieval quality and generation quality should be diagnosed separately with
   more detailed metrics than one final score.

   what it suggests for us:
   our current answer F1 is useful but not enough; later we should log whether
   the generated answer is:
   - supported by selected evidence
   - missing key facts
   - repeating unsupported filler

Current practical conclusion from the literature scan:

```text
The most promising next method for this project is not a heavier retriever.
It is:

selected chunks
-> query-oriented note compression
-> better evidence packing
-> final answer generation

This direction is more aligned with current answer-side failures than another
round of coarse/fine routing changes.
```

### Current Best Answer Mainline

After the answer-side ablations:

```text
turns + strict works best
chunk_turns hurts answer quality in the current implementation
```

Observed 25-case result:

```text
setting:
  selected_answer_context_mode = turns
  answer_prompt_style          = strict
  answer_evidence_order        = qk_then_time

result:
  avg selective-answer F1 = 23.1%
  avg answer F1 delta     = +4.2%
  selective >= full       = 72.0%
  ctx saving              = 86.0%
```

So the next answer-side improvement should not replace turns with another
evidence format. It should improve how selected turns are ordered and packed.

### Negative Answer-Side Results

Two answer-side variants were tested and should not be kept as the default
mainline:

```text
1. chunk_turns + strict
   avg selective-answer F1 = 19.2%
   selective >= full       = 48.0%

2. answer_aware + turns + strict
   avg selective-answer F1 = 20.4%
   selective >= full       = 60.0%
```

Interpretation:

```text
strict prompt itself is helpful
but replacing turns with chunk_turns hurts answer quality
and the current heuristic answer-aware rerank also hurts the best baseline
```

So the current best answer mainline remains:

```text
dynamic budget
+ qk_then_time evidence order
+ selected_answer_context_mode = turns
+ answer_prompt_style = strict
```

### Stage Closeout On 2026-06-16

What this stage established:

```text
1. lexical coarse routing is strong enough to keep as the frozen coarse mainline
2. topic-local Q-K chunk routing is a usable fine evidence selector
3. selected retrieval can save around 86% context on the 25-case check
4. answer quality can beat full-context answering if we keep:
     turns
     + strict prompt
     + qk_then_time ordering
5. more heuristic answer-side patches are now showing diminishing returns
```

Current best 25-case reference:

```text
setting:
  lexical coarse
  -> top-1 topic
  -> topic-local Q-K
  -> dynamic budget
  -> qk_then_time
  -> turns + strict

result:
  avg selective-answer F1 = 23.1%
  avg answer F1 delta     = +4.2%
  selective >= full       = 72.0%
  ctx saving              = 86.0%
```

### Phase Transition

This means the current heuristic tuning phase can be closed.

Next phase should move to a higher-level question instead of continuing to
micro-tune coarse/fine heuristics.

Priority direction A:

```text
make the QMSum line more explicit as distributed selective fetch simulation
```

Concretely:

```text
Current QMSum mainline:
  uses full local KV first
  then slices topic/chunk KV and estimates transfer cost

It does not yet instantiate physically separated per-node KV stores before
routing.

So the next systems step is:
  explicitly bucket topic/chunk KV into virtual nodes
  then route to those virtual nodes
  then report fetch/transfer accounting under that node layout
```

Priority direction B:

```text
upgrade answer generation from raw turns -> two-stage note compression
```

Concretely:

```text
selected turns
-> query-oriented notes
-> final answer
```

Why these are the right next phases:

```text
Direction A strengthens the distributed interpretation.
Direction B is the last answer-side method family that is still meaningfully
different from prompt/ordering heuristics.
```

Immediate project decision:

```text
Freeze the current best mainline as the reference baseline.
Do not keep chunk_turns or answer_aware as default.
Enter the next phase with:
  A. more explicit distributed-node simulation
  and/or
  B. note-compression answer interface
```

## INFOCOM-Oriented Note On 2026-06-16

We now keep a separate note for future submission-oriented planning:

```text
docs/infocom_submission_notes.md
```

Why this note exists:

```text
The current mainline is algorithmically stable enough
that the next planning question is no longer only
"which heuristic should we tune next?"

It is also:
"if this project later moves toward INFOCOM,
 what system metrics, comparisons, and distributed assumptions must be added?"
```

Current short conclusion:

```text
The project already has a usable selective-routing baseline.
What it still lacks for an INFOCOM-style story is mainly:
  1. harder system metrics
  2. clearer virtual-node distributed modeling
  3. stronger full-pull / selective-pull / remote-decode comparisons
```

## Virtual-Node Simulation Update On 2026-06-16

The QMSum mainline has now taken the first concrete step from:

```text
topic-level transfer accounting only
```

toward:

```text
virtual-node distributed-fetch simulation
```

What changed in the current code:

```text
1. topics are now mapped onto virtual nodes
2. each selected chunk now carries:
     transfer_topic_id
     transfer_node_id
3. transfer accounting is now kept in two views:
     topic-level
     virtual-node-level
4. logs now expose selected virtual node ids
```

What this means:

```text
The code still does not implement real multi-process distributed KV storage,
but it no longer treats "topic transfer" and "node transfer" as the same thing.

It now explicitly simulates:
  semantic topic shards
  -> packed into virtual nodes
  -> fetched selectively by node
```

Current boundary after this update:

```text
This is still a virtual-node shard simulation,
not a real network transport implementation.
```

So the next systems step after this update should be:

```text
convert the new node-level transfer accounting into:
  estimated bytes
  estimated fetch latency
  estimated TTFT
```

Update on 2026-06-16:

```text
That next step is now implemented in the mainline.

Each case now additionally records a simple system-cost estimate on top of
virtual-node transfer accounting:
  selected KV bytes / MiB
  full-pull KV bytes / MiB
  selected fetch latency
  full-pull fetch latency
  selected estimated TTFT
  full-pull estimated TTFT

Current cost model:
  kv_bytes = transferred_tokens * kv_bytes_per_token
  fetch_latency = bandwidth_time + per_node_rtt + per_segment_overhead
  estimated_ttft = routing_overhead + fetch_latency + decode_startup

This is still an explicit simulation model, not a real transport benchmark,
but it finally lets the project report system-style cost numbers instead of
only chunk/token counts.
```

## Mainline Audit And Bottleneck Update On 2026-06-17

We just used a tiny QMSum smoke run to audit whether the new system-cost layer
was telling a believable story.

The important outcome is:

```text
the code did expose a real system bottleneck,
but it also exposed several places where the prototype was easy to misread
or too optimistic.
```

### Problems Confirmed

1. `route_top_k` was previously not a true hard budget under `route_per_head`.

```text
Meaning:
  "top_k = 12" did not always mean "final selected chunks <= 12".

Risk:
  transfer reduction and system-cost numbers looked cleaner than the actual
  route budget semantics.
```

2. The old `qk` summary label was too easy to misinterpret.

```text
Meaning:
  in the hierarchical mainline, Q-K topic scores were only computed after
  lexical had already chosen the topic shortlist.

Risk:
  readers could mistakenly think `qk top-1` was an independent coarse router.
```

3. The first TTFT estimate exposed the real current bottleneck.

```text
Meaning:
  selective fetch itself was cheap,
  but online routing time dominated the total latency.

Observed pattern:
  selected fetch latency dropped a lot,
  but routing overhead was still tens of seconds in the research prototype.
```

### Immediate Fixes Already Applied

1. Hard-cap final selected chunks after per-head voting.
2. Rename the displayed `qk` summary signal to `qk_restricted`.
3. Stop charging the same routing time to both selective and full-pull TTFT.
4. Tighten routing-overhead timing so it better reflects online routing work.

### What The Latest Smoke Result Now Means

The current reading of the mainline is:

```text
Selective fetch is directionally valid:
  it can cut transferred KV and fetch latency a lot.

But the current online routing implementation is too expensive to serve as
the final systems story.
```

In other words:

```text
the bottleneck is no longer "can we save transferred KV?"

the bottleneck is now:
  can we make routing cheap enough
  that the saved transfer time is not swallowed by route computation?
```

### Next Phase

The next stage should stop chasing more routing heuristics and instead split the
system story into:

```text
1. what can be precomputed offline?
2. what must stay online?
3. how should full pull / selective pull / remote decode be compared fairly?
```

Concrete next priorities:

```text
A. separate offline-precomputable routing artifacts from online routing cost
B. build a cleaner full-pull baseline and then add remote-decode comparison
C. report route time, fetch time, and answer quality as separate dimensions
```

Update on 2026-06-17:

```text
Step A has now started in code.

The mainline explicitly builds doc-level routing artifacts once per meeting:
  topic -> virtual-node layout
  route candidates
  prototype lexical statistics

Then each query only performs the online query-dependent routing step.
```

What this changes conceptually:

```text
Before:
  every query re-did both route preparation and route decision work

Now:
  doc-level route preparation is separated from query-level online routing
```

What this does NOT solve yet:

```text
Q-K chunk scoring is still online and is still expected to dominate latency.

So this update is mainly:
  a cleaner systems decomposition
not yet
  the final latency fix
```

## Candidate Lexical Prefilter White-Box Note On 2026-06-18

This note records the new fine-stage prefilter in the most concrete way.

The new idea is not:

```text
replace Q-K chunk routing
```

It is:

```text
topic routing still picks a small set of topics first
-> inside those topics, run a cheap lexical candidate prefilter
-> only then run expensive Q-K scoring on the reduced candidate set
```

### A. Candidate lexical document

Code entry:

```text
build_candidate_lexical_document(...) in qmsum_mainline_routing.py
```

What it builds:

```text
one lightweight lexical "document" for each chunk candidate
```

Current content of that document:

```text
1. the candidate's turn speaker + turn content
2. the label text of the topic(s) that this candidate belongs to
```

Important clarification:

```text
this does not replace the old coarse topic label usage
```

Instead:

```text
topic label was already used in coarse topic routing
and is now also reused as a lexical hint for candidate prefilter
```

Why this exists:

```text
before this update,
chunk candidates had structural fields such as:
  candidate_id
  turn_idx
  topic_ids
  start_t / end_t

but they did not have a lightweight lexical representation
for cheap in-topic retrieval
```

### B. Candidate lexical stats / BM25-style mini-index

Code entry:

```text
build_candidate_lexical_stats(...) in qmsum_mainline_routing.py
```

What it precomputes:

```text
candidate_tf
candidate_lens
idf
avg_doc_len
```

Interpretation:

```text
after A writes one small lexical document per candidate,
B turns all of them into a reusable mini lexical index
for this meeting
```

Why this exists:

```text
when a query arrives,
the code can score many candidates quickly with BM25-style lexical matching
without running heavy KV slicing and Q-K scoring first
```

Current implementation boundary:

```text
this lexical mini-index is prebuilt once at the meeting level
for all route candidates in the document
```

So the current design is:

```text
doc-level prebuild
-> query-level reuse
```

This is useful when one meeting has multiple queries.

### C. Online candidate prefilter before Q-K

Code entry:

```text
score_mainline_topic_chunk(...) in qmsum_mainline_routing.py
```

Key code region:

```text
candidate_prefilter_mode
candidate_prefilter_pool_size
num_candidates_before_prefilter
candidate_prefilter_scores
candidate_prefilter_selected_ids
num_candidates_after_prefilter
```

What happens online for each query:

```text
1. lexical coarse routing chooses top topic(s)
2. collect all chunk candidates that belong to those selected topics
3. count how many candidates exist before prefilter
4. decide how many candidates to keep after prefilter
5. score those candidates with cheap lexical BM25-style matching
6. sort by lexical score
7. keep only the top prefilter pool
8. run expensive KV-based Q-K scoring only on the survivors
```

What is new compared with the old version:

```text
old fine stage:
  selected topic(s)
  -> Q-K score almost every candidate inside them

new fine stage:
  selected topic(s)
  -> cheap lexical prefilter
  -> Q-K score only the reduced candidate set
```

### Why this layer matters

The motivation is simple:

```text
Q-K chunk scoring is currently the expensive online step
```

Each candidate that reaches Q-K still triggers work like:

```text
split_kv(...)
stack_keys_from_kv(...)
qk_score_fn(...)
```

So this update tries to save online compute by reducing:

```text
candidates_to_score
```

before the heavy stage starts.

### What this layer does NOT yet mean

It does not mean:

```text
the system is already doing real remote lexical retrieval across distributed nodes
```

Current boundary remains:

```text
the lexical prefilter is built and used on the same experiment machine
as part of the selective-fetch simulation pipeline
```

So this is currently best described as:

```text
query-aware local prefiltering inside a simulated distributed selective-fetch setup
```

## Current Difficulty Snapshot On 2026-06-18

This note is meant to prevent the project from losing the main thread while
multiple sweeps are running.

### 1. The old turn-boundary bug is fixed, so current failures should now be read as real

What was fixed:

```text
prompt tokenization and turn_boundaries are now aligned under
add_special_tokens=False
```

Meaning:

```text
the earlier "selected evidence looks wrong" suspicion caused by token-span
misalignment is no longer the main explanation for current answer drops
```

So the current 5-doc / 5-query answer results should be treated as meaningful
signals, not as a token-boundary artifact.

### 2. The current answer-quality problem is concentrated on detail queries

Observed pattern from the recent 25-case check:

```text
summary-like queries are roughly fine
detail-like queries are the main source of selective-answer F1 drop
```

More concretely:

```text
summary subset:
  selective can match or beat full more often

detail subset:
  selective answer quality drops much more clearly
```

Interpretation:

```text
the current coarse topic routing is no longer the only issue
```

Even when the selected topic and selected turns are usable, the final answer for
detail questions can still degrade because:

```text
1. detail questions need tighter evidence completeness
2. the current chunk budget may be too small for narrow factual questions
3. selected evidence ordering / packing may still be suboptimal for detail QA
```

So the current answer-side bottleneck is:

```text
how to preserve detail-query evidence quality under selective context
```

### 3. Communication reduction is already strong, but TTFT is still dominated by routing overhead

Observed pattern:

```text
selected KV bytes and fetch latency drop a lot,
but selected estimated TTFT is still much worse than full-pull TTFT
because online routing time dominates the total
```

Interpretation:

```text
the bottleneck is no longer:
  can we reduce transferred KV?

the bottleneck is now:
  can we make online routing cheap enough that transfer savings matter end to end?
```

This means the current systems pressure is on:

```text
1. pushing more work into offline-precomputable artifacts
2. shrinking online Q-K candidate scoring cost
3. comparing full pull / selective pull fairly under the same accounting model
```

### 4. The virtual-node story is clearer than before, but the placement model is still simple

Important clarification:

```text
NUM_NODES = number of virtual transfer nodes
not
number of QMSum topics
```

Current implementation:

```text
QMSum topic_list -> topic nodes
topic nodes -> packed into virtual nodes
default packing mode = contiguous by topic order
```

This is useful for selective-fetch simulation, but it does NOT yet mean:

```text
topic placement is already optimized by semantic similarity or real system load
```

So the current communication/modeling gap is:

```text
we already simulate selective node fetch,
but we do not yet model a stronger topic-to-node placement policy
```

### 5. The project now has two active difficulties, not one

Current difficulty A:

```text
answer-side difficulty on detail queries
```

Current difficulty B:

```text
systems-side difficulty from large online routing overhead
```

This is the most important current summary:

```text
the project should not be read as "just tune routing accuracy more"
```

Instead, the current mainline has split into two coupled goals:

```text
Goal A:
  keep or recover detail-query answer quality under selective context

Goal B:
  reduce communication/TTFT bottlenecks by cutting online routing cost
```

### 6. Immediate working rule

Until the current sweep results are digested, avoid reopening too many side
branches at once.

Keep the workflow as:

```text
1. finish the current controlled sweep
2. separate detail-query quality changes from communication-cost changes
3. read each result as:
     answer quality
     routing overhead
     fetch latency
     estimated TTFT
4. only then decide whether the next patch should target:
     answer interface
     routing cost
     or virtual-node communication modeling
```

## Q-K Compute-Reduction Note On 2026-06-18

This note clarifies what the current mainline already does between coarse topic
routing and exact Q-K scoring, and what additional compute-reduction ideas are
now worth considering.

### 1. Important clarification: the current code already has one middle stage

The current active path is not:

```text
coarse topic routing
-> exact Q-K over all topic-local chunks
```

It is already:

```text
lexical coarse topic routing
-> topic_filter: keep only candidates inside selected topic(s)
-> candidate_prefilter: cheap lexical BM25-style candidate pruning
-> exact Q-K scoring on the survivors
```

So the current design already has:

```text
cheap lexical filter
before
expensive exact Q-K
```

This matters because the next systems question is no longer:

```text
should we add any middle stage at all?
```

It is:

```text
should the middle stage become more K-aware,
or should exact Q-K itself become cheaper?
```

### 2. Current implementation bottleneck shape

The expensive part of the current fine stage is still:

```text
for each surviving candidate:
  split_kv(...)
  stack_keys_from_kv(...)
  qk_score_fn(...)
```

So the current cost is not just:

```text
how many final chunks are selected
```

It is more directly:

```text
how many candidates still reach exact per-candidate Q-K scoring
and how fragmented that scoring loop is
```

### 3. Immediate low-risk direction: shrink the exact-Q-K pool further

This is what the new prefilter-budget sweep is meant to test.

The idea is simple:

```text
keep the current lexical candidate prefilter,
but tune factor / min_keep / max_keep
to reduce the number of candidates that still reach exact Q-K
```

This is the most conservative next step because it does not change:

```text
1. the coarse topic routing logic
2. the final exact-Q-K scoring definition
3. the answer-side interface
```

It only asks:

```text
can we save online routing cost by making the existing prefilter tighter?
```

### 4. Next algorithmic direction A: batch exact Q-K scoring

The current exact-Q-K stage is very fragmented.

A natural next step is:

```text
collect a batch of candidate K tensors
-> score them together
instead of
candidate-by-candidate Python loops
```

This is attractive because:

```text
1. it does not change the routing semantics
2. it reduces Python overhead and repeated small kernel launches
3. it is compatible with the current candidate structure
```

So this is the cleanest next code change if the current sweep confirms that:

```text
exact Q-K remains the dominant online cost even after prefilter tightening
```

### 5. Next algorithmic direction B: K-summary upper-bound pruning

This is the most promising new idea for the current codebase.

Motivation:

```text
the current middle stage is lexical-only
```

What is still missing is:

```text
a cheap K-aware stage
before exact token-level Q-K
```

The closest reusable inspiration comes from Quest-style metadata ideas:

```text
precompute small K summaries for each candidate or segment
then use them online to estimate whether exact Q-K is worth running
```

The most concrete version is:

```text
offline per candidate:
  store min(K), max(K), or other lightweight K summaries

online per query:
  compute a cheap upper bound or proxy score from Q and the summaries
  rank or prune candidates with that estimate
  run exact Q-K only on the survivors
```

The key point is:

```text
this would be the first middle stage that is not only lexical,
but still much cheaper than exact token-level Q-K
```

### 6. Next algorithmic direction C: coarse-to-fine K-aware scoring inside a topic

Another plausible direction is:

```text
selected topic
-> cheap K-aware score on larger contiguous segments
-> exact Q-K only inside the winning segments
```

This is different from the current lexical prefilter because the gate would be:

```text
K-aware
not just lexical
```

Examples:

```text
1. merged contiguous segment first, fine chunk second
2. turn-level K summary first, chunk-level exact Q-K second
3. topic-local coarse segment budget before exact chunk budget
```

This direction matches the current concern that:

```text
exact Q-K may be too fragmented at the current chunk granularity
```

### 7. Next algorithmic direction D: early exit over layers / heads

The current scoring path uses multiple layers and then aggregates.

A possible compute-saving strategy is:

```text
score a small subset of layers or heads first
-> maintain a running partial score or upper bound
-> stop early for candidates that are already clearly non-competitive
```

This is especially relevant because a prior Quest-style note suggested that:

```text
some shallow or edge layers may be weak for sparsity discrimination
```

So an ablation worth considering later is:

```text
which scoring layers actually contribute useful ranking signal,
and which layers mostly add cost
```

### 8. Next algorithmic direction E: threshold / budget stop instead of fixed exact top-k

The current logic still ultimately behaves like:

```text
score survivors exactly
-> keep a fixed top-k
```

A later refinement could be:

```text
score in descending proxy order
-> stop once a score threshold or budget criterion is satisfied
```

This may help when:

```text
some queries have only a very small number of clearly strong candidates
```

### 9. Current recommended order

Given the current code state, the safest order is:

```text
Step 1:
  run the new prefilter-budget sweep

Step 2:
  if exact Q-K still dominates, implement batched exact-Q-K scoring

Step 3:
  add one lightweight K-summary pruning layer
  (min/max or other cheap candidate metadata)

Step 4:
  only then consider more structural changes such as
  coarse K-aware segment gating or layer/head early-exit rules
```

The main reason for this order is:

```text
it moves from:
  smaller execution change
to:
  larger algorithmic change

while staying focused on the confirmed systems bottleneck:
  online Q-K routing cost
```

## Batched Exact-QK Update On 2026-06-19

This note records what changed after adding batched exact-QK scoring to the
current QMSum mainline.

### 1. The batching change is a real systems win, not just a cleaner implementation

Cloud-side compare now shows that increasing `qk_score_batch_size` reduces
online routing cost a lot while keeping the routing semantics unchanged.

Observed pattern:

```text
batch_8  -> avg_qk_scoring_ms about 2404
         -> avg_routing_ms about 2615
         -> avg_selected_ttft_ms about 2647

batch_32 -> avg_qk_scoring_ms about 746
         -> avg_routing_ms about 964
         -> avg_selected_ttft_ms about 997

batch_16 -> avg_qk_scoring_ms about 1286
         -> avg_routing_ms about 1498
         -> avg_selected_ttft_ms about 1531

batch_64 -> avg_qk_scoring_ms about 509
         -> avg_routing_ms about 727
         -> avg_selected_ttft_ms about 759
```

Meaning:

```text
the Python-loop / fragmented-kernel part of exact Q-K was real,
and batching removed a large part of that overhead
without changing the selected-answer quality
```

### 2. Current accepted execution default

For the current implementation, the best execution-side default is now:

```text
QK_SCORE_BATCH_SIZE=64
```

This should be treated as the current mainline execution setting unless a later
memory-pressure issue forces us to back off.

### 3. Batching helped a lot, but the end-to-end TTFT story is still not finished

Even after batching, the selective path is still slower than the simulated full
pull baseline in TTFT terms.

Current reading:

```text
selected estimated TTFT: about 759 ms
full-pull estimated TTFT: about 489 ms
```

Interpretation:

```text
communication reduction is already real,
but online routing is still expensive enough
that it can swallow the saved fetch time
```

So the systems bottleneck has now shifted again:

```text
not "is exact Q-K too fragmented?" alone
but
"what repeated work still happens around exact Q-K for every query?"
```

### 4. The next highest-value optimization target is repeated candidate-K preparation

The current scoring path still repeatedly pays for:

```text
split_kv(...)
stack_keys_from_kv(...)
```

inside the per-query candidate scoring flow.

That means the next strong engineering direction is:

```text
doc-level candidate stacked-K reuse
```

More concretely:

```text
build candidate K tensors once for a meeting document
-> reuse them across multiple queries on that same document
-> keep the exact scoring definition unchanged
```

Why this is the current priority:

```text
1. it targets repeated online work directly
2. it does not require changing the coarse/fine routing semantics
3. it should compose naturally with the current batch-QK path
```

### 5. Current practical working rule

For the next round of optimization, keep the answer-side mainline stable and
focus on routing-side execution cost first.

So the recommended order is:

```text
1. keep the current answer-side setting stable
   dynamic_detail_top_k=12

2. keep the current execution-side setting stable
   QK_SCORE_BATCH_SIZE=64

3. optimize repeated candidate-K preparation / reuse

4. only after that, revisit more aggressive K-aware pruning ideas
   such as K-summary bounds or coarse K-aware segment gates
```

## Adaptive Candidate Prefilter Skip On 2026-06-19

The lexical candidate prefilter is useful only when it removes enough chunk
candidates before exact Q-K scoring.

One observed case was:

```text
num_candidates_before_prefilter = 100
candidate_prefilter_pool_size   = 96
num_candidates_after_prefilter  = 96
```

That means the prefilter paid an extra lexical scoring step but removed only
four candidates. This is not a strong systems win, and it makes timing harder to
interpret because the prefilter is effectively a near no-op.

The mainline now has:

```text
route_candidate_prefilter_min_prune_ratio
```

Implemented default after validation:

```text
0.0
```

Meaning:

```text
if this value is set above 0.0 and lexical prefilter would prune too few candidates,
skip lexical prefilter for that query/topic pool
and send the full topic-local candidate set to exact Q-K
```

Concrete example:

```text
before candidates = 100
requested pool    = 96
prune ratio       = 4%
min prune ratio   = 20%

=> candidate_prefilter_mode = lexical_skipped_low_prune
=> candidate_prefilter_pool_size = 100
=> exact Q-K scores all 100 candidates
```

This does not change exact Q-K scoring or final chunk scoring semantics. It only
prevents a weak lexical prefilter from running when it cannot materially reduce
the expensive Q-K candidate pool.

New output fields:

```text
candidate_prefilter_requested_pool_size
candidate_prefilter_prune_ratio
candidate_prefilter_min_prune_ratio
candidate_prefilter_skip_reason
```

Next validation:

```text
compare min_prune_ratio=0.0 vs 0.2 on the same docs/queries
```

Observed on docs 5:20, max 5 queries/doc:

```text
min_prune=0.0:
  selective F1 about 17.0%
  exact Q-K about 636 ms
  routing about 669 ms
  selected TTFT about 705 ms

min_prune=0.2:
  selective F1 about 17.1%
  exact Q-K about 648-653 ms
  routing about 682-687 ms
  selected TTFT about 718-723 ms
```

Reading:

```text
the adaptive skip did remove near-no-op prefilter work,
but that work was only about 0.1 ms/query.

Skipping it made some queries score slightly more Q-K candidates,
so total routing became slower.
```

Decision:

```text
keep route_candidate_prefilter_min_prune_ratio=0.0 as the mainline default.
Keep the parameter for explicit ablations only.
```

## Coarse Segment Gate Before Exact Q-K On 2026-06-19

The min-prune experiment showed that saving the tiny lexical prefilter time is
not the right target. The real target remains:

```text
reduce how many candidates enter exact Q-K
```

New optional gate:

```text
route_coarse_segment_gate=lexical
```

Where it runs:

```text
lexical coarse topic routing
-> optional lexical candidate prefilter
-> optional coarse segment gate
-> exact Q-K on the remaining chunks
```

The gate groups neighboring candidate chunks into coarse segments, scores each
segment using the already available lexical candidate scores, keeps the strongest
segments, and only sends chunks inside those kept segments into exact Q-K.

Main knobs:

```text
ROUTE_COARSE_SEGMENT_GATE=lexical
ROUTE_COARSE_SEGMENT_SIZE=4
ROUTE_COARSE_SEGMENT_KEEP_RATIO=0.5
ROUTE_COARSE_SEGMENT_MIN_KEEP=48
ROUTE_COARSE_SEGMENT_MAX_KEEP=0
```

Important: this is not enabled by default. It is an ablation path because it can
reduce Q-K time but may hurt selected evidence quality if lexical segment scores
miss important chunks.

New output fields:

```text
coarse_segment_gate_mode
coarse_segment_gate_before
coarse_segment_gate_after
coarse_segment_gate_prune_ratio
coarse_segment_gate_ms
```

The first validation should compare:

```text
baseline:     ROUTE_COARSE_SEGMENT_GATE=none
coarse gate:  ROUTE_COARSE_SEGMENT_GATE=lexical
```

Decision rule:

```text
keep exploring this direction only if:
  Q-K time drops materially
  routing/TTFT drops materially
  selected-answer F1 and turn recall do not collapse
```

Expected reading:

```text
0.0 keeps the old always-run behavior
0.2 skips near-no-op lexical prefilter cases

If selected F1 is unchanged and candidate_prefilter_ms goes down,
keep 0.2 as the cleaner default.

If qk_scoring_ms rises too much because skipped cases score many more chunks,
lower the threshold or keep lexical prefilter always on for larger candidate pools.
```

## Coarse Segment Gate Budget Sweep On 2026-06-20

First coarse-gate validation on docs 5:20, max 5 queries/doc:

```text
baseline:
  selected-answer F1 about 0.1703
  Q-K scoring about 426.6 ms
  selected TTFT about 484.2 ms

coarse gate lexical, segment_size=4, keep_ratio=0.5, min_keep=48:
  selected-answer F1 about 0.1643
  Q-K scoring about 262.0 ms
  selected TTFT about 315.9 ms
```

Reading:

```text
This is the first setting where the selective path becomes clearly useful for
system time: exact Q-K and selected TTFT both drop a lot.

The answer F1 drop is small but real, so the next question is not whether
coarse gate works at all. The next question is whether a gentler gate can
recover quality while keeping most of the Q-K reduction.
```

Next script:

```text
scripts_qmsum/run_qmsum_coarse_segment_gate_budget_sweep_4gpu.sh
```

Default sweep:

```text
baseline none
segment_size=4 keep_ratio=0.50 min_keep=48
segment_size=4 keep_ratio=0.65 min_keep=48
segment_size=4 keep_ratio=0.75 min_keep=48
segment_size=4 keep_ratio=0.65 min_keep=64
segment_size=4 keep_ratio=0.75 min_keep=64
```

Decision rule:

```text
Prefer the gentlest setting whose selected-answer F1 is close to baseline
while steady_qk_ms / steady_selected_ttft_ms remain much lower than baseline.

Use steady_* columns when judging timing because they exclude first-query
warmup effects.
```

## Larger Coarse Gate Validation On 2026-06-20

Validation command:

```text
START_DOC=5 END_DOC=60 MAX_QUERIES=5
RUN_BASELINE=1
GATE_CASES="4:0.65:64:0 4:0.75:64:0"
```

Observed on 269 query cases:

```text
baseline:
  full F1:             0.1978
  selected F1:         0.1495
  selected-full gap:  -0.0483
  steady Q-K:          421.53 ms
  steady selected TTFT:476.25 ms

s4 r0.65 m64:
  full F1:             0.1978
  selected F1:         0.1515
  selected-full gap:  -0.0463
  steady Q-K:          267.03 ms
  steady selected TTFT:320.69 ms

s4 r0.75 m64:
  full F1:             0.1978
  selected F1:         0.1529
  selected-full gap:  -0.0449
  steady Q-K:          298.09 ms
  steady selected TTFT:351.64 ms
```

Reading:

```text
Coarse gate is still beneficial at larger scale.

The gate itself is not the main source of answer-quality loss here. Both gate
settings slightly improve selected F1 compared with the no-gate selected path.

The remaining problem is the larger selected-vs-full gap, about 4.5-4.8 F1
points on this slice. This should be debugged as a quality gap problem, not as
a gate-only problem.
```

Current default candidate:

```text
route_coarse_segment_gate=lexical
route_coarse_segment_size=4
route_coarse_segment_keep_ratio=0.65
route_coarse_segment_min_keep=64
```

Reason:

```text
s4 r0.75 m64 has slightly better selected F1, but s4 r0.65 m64 gives much lower
steady Q-K / TTFT and still improves selected F1 over baseline.
```

Next analysis tool:

```text
scripts_qmsum/run_qmsum_bad_case_analysis.sh
```

It reads saved case-summary TSVs and answer logs, then produces:

```text
worst_selected_vs_full.tsv
worst_selected_vs_oracle.tsv
gate_regressions_vs_baseline.tsv
bad_case_report.md
```

Purpose:

```text
Identify whether the selected-full gap mostly comes from:
  topic routing misses
  evidence/chunk routing misses
  answer generation using the selected evidence poorly
  rare gate regressions
```

## Current Mainline Closeout Checkpoint On 2026-06-21

We stop the current-mainline closeout at docs `0:30`, max `5` queries per
doc. No need to backfill `30:40` for now.

The run used:

```text
--mainline_profile current
docs 0:30
MAX_QUERIES=5
n=148 query cases
```

Important command note:

```text
The intended smoke command had `GPU_ID=0 \ ` with a trailing space after the
backslash. Bash therefore did not pass START_DOC=0 END_DOC=3 MAX_QUERIES=2
into the script, and the script fell back to its default docs 0:30 q5.
```

Observed checkpoint:

```text
full-answer F1:      18.0%
selective-answer F1: 16.8%
oracle-answer F1:    20.9%
selected-full delta: -1.2%
ctx token saving:    94.8%

selected KV:         49.7 MiB
full KV:             1638.6 MiB
KV reduction:        96.2%
selected fetch:      19.12 ms
full fetch:          554.33 ms

online routing:      291.47 ms
Q-K model:           271.00 ms
exact Q-K:           271.33 ms
Q-K total stage:     285.62 ms
selected TTFT:       325.59 ms
full TTFT:           569.33 ms
TTFT reduction:      29.3%
```

Decision:

```text
Treat this as the temporary current-mainline baseline.

Do not keep extending top2 rescue or neighbor expansion. Both have already
failed to improve selected F1 / turn recall on 0:30 q5.

Next work should target the real bottleneck: online exact Q-K cost. Move toward
cheap offline descriptors / query-time descriptor scoring / exact Q-K as a
teacher or small-candidate reranker.
```

## Timing Accounting Revision On 2026-06-21

This note supersedes the timing numbers in the previous closeout block, but does
not delete them because they are useful history.

Latest validated split run:

```text
command family:
  scripts_qmsum/run_qmsum_current_mainline_4gpu_split.sh

docs:
  0:30

max queries per doc:
  5

query cases:
  148

output shards:
  outputs/qmsum_case_summary_N4_0_8_current_mainline_0_8_q5.tsv
  outputs/qmsum_case_summary_N4_8_16_current_mainline_8_16_q5.tsv
  outputs/qmsum_case_summary_N4_16_24_current_mainline_16_24_q5.tsv
  outputs/qmsum_case_summary_N4_24_30_current_mainline_24_30_q5.tsv
```

Quality checkpoint:

```text
full answer F1:       18.04%
selected answer F1:   16.79%
oracle answer F1:     20.94%
selected-full delta:  -1.25 F1 points
selected >= full:     45.27%

selected bad output:  8.11%
full bad output:      4.05%
oracle bad output:    3.38%
```

Timing checkpoint:

```text
routing wall clock:          78.79 ms
system-accounted routing:    64.21 ms
simulator key slicing excl.:  14.58 ms

query-Q prepare:             27.40 ms
Q-K model:                   29.95 ms
Q-K scoring:                 57.71 ms
Q-K total stage, wall/debug: 72.29 ms

selected fetch:              19.12 ms
full fetch:                  554.33 ms
selected TTFT:               98.33 ms
full TTFT:                   569.33 ms
TTFT saving:                 78.67%
ctx token saving:            94.78%
```

Important interpretation:

```text
selected TTFT is an estimated system TTFT:

  selected_TTFT
    = system_accounted_routing
    + selected_fetch_latency
    + decode_startup

  98.33 ms
    ~= 64.21 ms
     + 19.12 ms
     + 15.00 ms

full_TTFT
    = full_fetch_latency
    + decode_startup

  569.33 ms
    ~= 554.33 ms
     + 15.00 ms
```

Why TTFT dropped so much:

```text
1. Query-Q cache became part of the current mainline.

   Old exact Q-K scoring repeatedly ran the query-side forward/projection for
   each candidate batch. The query hidden states and Q projections depend only
   on the query and scoring layers, not on which candidate chunk is being
   scored. Therefore they can be computed once per query and reused across all
   candidate batches.

   This changed the expensive online Q-K model part from roughly hundreds of
   milliseconds to about 30 ms on the 0:30 q5 checkpoint.

2. Candidate key preparation is now separated from system TTFT.

   candidate_key_prepare_ms is the local simulator slicing full prefilled KV:

     full KV tensor
       -> slice candidate span
       -> stack keys into scorer format

   In a real distributed KV system, the remote node/chunk store already owns
   these KV blocks. The router should not pay the cost of reconstructing
   candidate keys from one monolithic local full KV tensor. Therefore this time
   is still reported as routing wall/debug time, but excluded from the
   system-accounted TTFT.

   Current accounting:

     routing wall clock       = 78.79 ms
     simulator key excluded   = 14.58 ms
     system routing           = 64.21 ms

3. Selected fetch is much smaller than full fetch.

   selected KV: about 49.7 MiB
   full KV:     about 1638.6 MiB

   Under the current 25 Gbps bandwidth model, this makes selected fetch about
   19 ms while full fetch is about 554 ms.
```

Is this reasonable?

```text
Reasonable as a research-system estimate:

  - Query-Q cache is algorithmically valid if the cached Q tensors are computed
    with the same hidden-state/layer convention as the old scorer.
  - Excluding candidate_key_prepare_ms is reasonable for a distributed-KV
    system model because it is a local simulator artifact.
  - Keeping routing_wall_clock_ms and routing_simulator_excluded_ms in the logs
    is necessary so we do not hide this accounting choice.

Not yet a production TTFT claim:

  - It is not a real multi-machine serving measurement.
  - It assumes serial routing -> fetch -> decode startup.
  - It does not model network queueing, scheduling contention, real RPC
    overheads, decompression, or full end-to-end generation throughput.
  - The full baseline also assumes the full KV already exists and only needs to
    be fetched; this is a fair KV-transfer comparison, not a full prefill
    comparison.
```

Current bottleneck after this revision:

```text
The main timing bottleneck is no longer giant repeated Q-K model cost.
The current quality bottlenecks are:

  1. selected bad output rate is higher than full
  2. topic/turn miss cases still cause most of the selected-full F1 gap

Do not interpret the 98 ms selected TTFT as final proof. Interpret it as:

  the routing-time accounting is now plausible enough to move attention back to
  quality failures and bad-output prevention.
```

## Candidate Survival Diagnosis On 2026-06-21

Motivation:

```text
The 0:20 current-mainline split run completed successfully, but docs 0:5 showed
a large local selected-full F1 gap. One representative case had the correct
coarse topic, yet selected turns were shifted away from the annotated relevant
turns.

This means the next question is not just "which parameter should be bigger".
The first question is:

  where do gold-turn candidates disappear?

Possible failure points:

  candidate_build
  -> topic_filter
  -> lexical candidate_prefilter
  -> dynamic_candidate_pool
  -> coarse_segment_gate
  -> Q-K scored set
  -> final Q-K selected top-k
```

Code change:

```text
qmsum_mainline_routing.py now records candidate ids surviving each funnel stage
and full Q-K ranks for scored candidates.

qmsum_mainline.py now builds relevant_candidate_survival during evaluation only.
This uses QMSum gold relevant turns after routing has finished, so it does not
affect online routing decisions or selected chunks.

qmsum_output.py writes survival fields into the case TSV and answer JSONL.
qmsum_eval.py prints a Relevant candidate survival summary.
```

New fields to inspect in case TSV:

```text
survival_failure_stage
survival_first_drop_stage
survival_first_zero_stage
survival_topic_filter_turn_recall
survival_prefilter_turn_recall
survival_dynamic_pool_turn_recall
survival_coarse_gate_turn_recall
survival_qk_scored_turn_recall
survival_qk_selected_turn_recall
```

Interpretation:

```text
If recall drops sharply at candidate_prefilter, the lexical prefilter is too
aggressive or too brittle.

If recall drops sharply at dynamic_candidate_pool, the query-type fixed budget
is likely hurting quality and should be disabled or replaced by an adaptive
score/latency policy.

If recall stays high until Q-K scored but drops at qk_selected, candidate
control is not the main culprit; exact Q-K ranking or final top-k selection is.

If recall already drops at topic_filter, the coarse topic router is still the
main quality bottleneck for that case.
```

## Quality Guard Profile On 2026-06-22

Motivation:

```text
The candidate-survival run showed that the current fast path saves TTFT, but
some relevant-turn candidates disappear before exact Q-K:

  topic filter -> lexical candidate prefilter -> dynamic candidate pool
  -> coarse segment gate -> exact Q-K -> final top-k

The largest early-risk stages were lexical prefilter and dynamic candidate
pool. This validates the concern that fixed candidate caps such as 48/56/96 may
not generalize across documents or datasets.
```

Code change:

```text
qmsum_mainline_config.py adds mainline_profile=quality_guard.

quality_guard keeps the same high-level system path as current:

  lexical topic routing
  -> lexical candidate prefilter
  -> coarse segment gate
  -> batched exact Q-K
  -> selected answer evaluation and TTFT accounting

But it is intentionally more conservative before exact Q-K:

  dynamic_candidate_pool_budget = False
  route_candidate_prefilter_factor = 12
  route_candidate_prefilter_min_keep = 96
  route_candidate_prefilter_max_keep = 256
  route_candidate_prefilter_keep_ratio = 0.85
  route_candidate_prefilter_min_prune_ratio = 0.20
  route_coarse_segment_keep_ratio = 0.85
  route_coarse_segment_min_keep = 128

qmsum_mainline_routing.py now supports route_candidate_prefilter_keep_ratio, so
lexical prefilter can be bounded by a percentage of the current candidate pool
instead of only fixed keep counts.

run_qmsum_current_mainline.sh and run_qmsum_current_mainline_4gpu_split.sh now
accept MAINLINE_PROFILE, so current and quality_guard can be compared without
editing scripts.
```

How to interpret:

```text
This is not the final routing policy. It is a quality-recovery profile.

If quality_guard improves selected F1 and survival recall while keeping TTFT far
below full, the next step is to replace fixed candidate control with a cleaner
adaptive policy.

If survival recall improves but selected F1 does not, then the main failure has
moved from early pruning to exact Q-K ranking/final top-k evidence selection.
```
