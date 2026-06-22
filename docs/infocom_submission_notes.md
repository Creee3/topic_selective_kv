# INFOCOM Submission Notes

## Purpose

This note records the current understanding of how the `topic_selective_kv`
project should be reframed and expanded if we later target an INFOCOM-style
submission.

The goal is not to change the current QMSum mainline immediately.
The goal is to keep one stable reference note for:

```text
paper framing
evaluation gaps
system metrics to add
future experiment priorities
```

## Current Best Mainline

Current best validated QMSum answer setting:

```text
lexical coarse topic routing
-> top-1 topic
-> topic-local Q-K chunk routing
-> dynamic chunk budget
-> qk_then_time answer evidence order
-> selected turns
-> strict answer prompt
```

Observed 25-case result:

```text
avg selective-answer F1 = 23.1%
avg answer F1 delta     = +4.2%
selective >= full       = 72.0%
ctx saving              = 86.0%
```

This means the current project already supports one useful claim:

```text
selective evidence transfer can preserve or even improve answer quality
while reducing transferred context substantially
```

## Current Boundary

This boundary must be stated honestly in any future paper draft:

```text
The current QMSum code is a distributed selective-fetch simulation,
not yet a full physically separated distributed serving system.
```

Current implementation behavior:

```text
1. obtain full local meeting/KV context
2. slice it into topic/chunk segments
3. simulate selective fetch and transfer accounting on top of those slices
```

So the current contribution is closer to:

```text
routing policy validation
+ evidence quality evaluation
+ transfer reduction simulation
```

and not yet:

```text
real multi-node KV deployment
real transport implementation
real online serving system
```

## INFOCOM-Oriented Reframing

If this work is later written for INFOCOM, it should not be framed as:

```text
selective retrieval improved answer F1
```

It should be framed more like:

```text
query-aware selective KV fetch for distributed LLM serving
```

or:

```text
reducing cross-node KV transfer while preserving answer quality
```

The key story should become:

```text
In a distributed LLM / distributed KV setting,
can query-aware routing reduce transfer and first-token latency
without hurting final answer quality too much?
```

## What INFOCOM-Style Evaluation Still Needs

The current project already has:

- topic hit metrics
- turn recall / precision / F1
- answer F1
- context saving

But an INFOCOM-style paper will also need stronger system-facing metrics.

### Must-add system metrics

- estimated transferred bytes
- transfer reduction ratio
- routing latency / routing overhead
- estimated fetch latency
- estimated TTFT
- p50 / p95 / p99 latency or TTFT
- request throughput
- token throughput
- overhead breakdown:
  - routing time
  - packing / coalescing time
  - transfer time
  - decode time

### Must-add system comparisons

- `full pull`
- `selective pull`
- `remote decode` or `remote query execution`

This is especially important because it matches the advisor discussion:

```text
Should we pull remote KV back,
or ship the query / decoding work outward instead?
```

### Must-add scaling / sensitivity studies

- bandwidth sensitivity
- node-count sensitivity
- chunk-budget sensitivity
- coalescing on/off
- quality under fixed transfer budget
- transfer under fixed quality target

## Suggested Core Figures

At minimum, future INFOCOM-oriented evaluation should aim to produce:

1. `answer F1 vs transferred bytes`
2. `estimated TTFT vs bandwidth`
3. `quality vs node count`

Very useful additional curves:

- `transfer reduction vs bandwidth`
- `answer F1 vs route_top_k`
- `selected >= full ratio vs transfer budget`
- `routing overhead vs total saved transfer`

## Suggested Experimental Structure

The evaluation section should eventually be organized more like a systems paper:

### 1. Routing / retrieval quality

- topic top-1 / top-2 hit
- selected-turn recall / precision / F1
- answer support coverage

### 2. Answer quality

- full-answer F1
- selective-answer F1
- answer F1 delta
- selective >= full ratio

### 3. System efficiency

- transferred chunks
- transferred tokens
- estimated transferred bytes
- routing overhead
- estimated TTFT
- throughput

### 4. Scalability / sensitivity

- bandwidth
- node count
- route budget
- coalescing gain

### 5. Ablations

- coarse routing choice
- fine routing budget
- transfer packing choices
- answer interface choices

## Reference Repos To Borrow From

### CacheGen

Most useful borrowable evaluation style:

- TTFT
- bandwidth sensitivity
- overhead breakdown
- storage / transfer cost
- speedup ratio

Relevant local files:

- `reference_repos/CacheGen-main/measure_ttft.py`
- `reference_repos/CacheGen-main/fig11_bandwidth.py`
- `reference_repos/CacheGen-main/fig14_overhead.py`

### Quest

Most useful borrowable evaluation style:

- quality + efficiency together
- Passkey
- LongBench
- PG-19 perplexity
- self-attention speedup
- end-to-end speedup

Relevant local file:

- `reference_repos/Quest-main/README.md`

### InfiniGen

Most useful borrowable evaluation style:

- split experiments into `accuracy` and `speedup`

Relevant local file:

- `reference_repos/InfiniGen-main/README.md`

### KVDirect

Most useful borrowable evaluation style:

- latency
- throughput
- serving benchmark outputs
- transport-aware benchmarking mindset

Relevant local files:

- `reference_repos/KVDirect-main/benchmarks/benchmark_latency.py`
- `reference_repos/KVDirect-main/benchmarks/benchmark_throughput.py`
- `reference_repos/KVDirect-main/benchmarks/benchmark_serving.py`

## Current Project Weaknesses Relative To INFOCOM

At the current stage, the main gaps are:

1. too little system-side evaluation
2. no explicit virtual-node storage model in the active QMSum line
3. no direct `full pull vs selective pull vs remote decode` comparison yet
4. not enough latency / throughput / bandwidth-facing metrics
5. current results are still mainly small-to-medium-scale research validation

These gaps do not invalidate the current work.
They simply mean:

```text
the current project is algorithmically promising,
but not yet fully packaged as a networking/systems paper
```

## Immediate Priority If We Move Toward Submission

Recommended order:

1. freeze the current best QMSum mainline as the reference baseline
2. add explicit virtual-node distributed-fetch simulation
3. convert transfer accounting into estimated bytes / latency / TTFT

Status update on 2026-06-16:

```text
This item has now moved from "planned" to "implemented".

The current QMSum mainline already exports:
  virtual-node transfer accounting
  estimated selected/full KV bytes
  estimated selected/full fetch latency
  estimated selected/full TTFT

The present model is still analytical rather than measured on a real
distributed runtime, so the paper should describe these as:
  estimated system cost under a virtual-node selective-fetch model
```
4. add `full pull vs selective pull vs remote decode`
5. add bandwidth and node-count sensitivity
6. only after that, consider larger benchmark expansion

Current implementation status:

```text
Step 2 has started:
  the code now supports topic -> virtual node mapping
  and keeps node-level transfer accounting beside topic-level accounting.

What is still missing is the latency/TTFT layer on top of that accounting.
```

## Useful External Pointers

- INFOCOM 2025 DBLP proceedings:
  [https://dblp.org/db/conf/infocom/infocom2025.html](https://dblp.org/db/conf/infocom/infocom2025.html)
- Online Context Caching for Distributed Large Language Models Serving:
  [https://dblp.org/rec/conf/infocom/GaoHYL0W25](https://dblp.org/rec/conf/infocom/GaoHYL0W25)
- Mell: Memory-Efficient Large Language Model Serving via Multi-GPU KV Cache Management:
  [https://dblp.org/rec/conf/infocom/LiuHLC025](https://dblp.org/rec/conf/infocom/LiuHLC025)
- AdaRAG: Adaptive Optimization for Retrieval Augmented Generation with Multilevel Retrievers at the Edge:
  [https://dblp.org/rec/conf/infocom/OuyangHZZWL025](https://dblp.org/rec/conf/infocom/OuyangHZZWL025)

## Short Takeaway

The current project already has a viable algorithmic mainline.

The next transformation is:

```text
from:
  can selective routing keep useful evidence?

to:
  can selective distributed fetch reduce transfer and latency
  while preserving answer quality?
```

## System-Risk Note On 2026-06-17

The first system-cost smoke check surfaced an important paper-positioning fact:

```text
the current prototype already shows large transfer savings,
but the online routing path is still too expensive to support a strong
end-to-end latency story.
```

This is not a failure of the selective-fetch idea.
It means the paper should not yet claim:

```text
our current implementation already improves end-to-end TTFT
```

What can be claimed more safely now:

```text
1. selective KV fetch can greatly reduce estimated transferred KV
2. selective KV fetch can greatly reduce estimated fetch latency
3. the current prototype exposes online routing cost as the next main bottleneck
```

Therefore the next INFOCOM-oriented system step is:

```text
turn routing itself into a systems question:
  which parts can be cached or precomputed offline,
  and which parts must stay online?
```

This will make the project story much stronger than continuing to add more
small routing heuristics.

Status update on 2026-06-17:

```text
The mainline now explicitly separates:
  doc-level routing artifact preparation
from:
  query-level online routing

This is a small but important systems-design step because it turns
"routing cost" from one opaque number into:
  offline-precomputable cost
  online decision cost
```
