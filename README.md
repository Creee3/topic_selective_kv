# Topic-Selective KV Routing

This directory contains the current prototype for **Q-K attention based selective KV cache routing**.

The main research question is:

```text
When KV cache is distributed across nodes, can the model's own Q-K attention signal decide which nodes/chunks are worth fetching for a new query?
```

## Current Position

The project is closest to **Quest**, but changes the target problem:

```text
Quest:
  Query-aware selection of local KV pages during decode.

This work:
  Query-aware routing/fetching of distributed KV chunks or nodes.
```

CacheGen is treated as a compression backend after routing has decided what to transfer.

## Current Scope Alignment

The current QMSum line should be described carefully:

```text
This is a routing simulation / prototype,
not yet a full distributed serving system.
```

Current agreed assumptions:

```text
1. top-level topic structure can be given by dataset labels
2. turn -> topic assignment can also rely on dataset annotations
3. current focus is not automatic topic discovery
4. current focus is:
   query routing -> fine-grained evidence selection -> transfer reduction -> later answer-quality comparison
```

In other words, the active QMSum mainline is:

```text
query
-> coarse routing over labeled topics
-> Q-K fine routing over chunks inside selected topics
-> compare selected evidence with relevant spans
-> estimate transfer cost
-> compare answer quality
```

This scope is important because it matches the current advisor discussion:

```text
First assume semantic nodes already exist,
then study whether selective routing can reduce traffic
without hurting final answer quality too much.
```

Important boundary for the current QMSum line:

```text
The current code does not yet pre-shard KV into physically separated node
stores before routing.

It first obtains full local KV for one meeting,
then slices topic/chunk KV segments and uses those slices to simulate
selective distributed fetch and transfer accounting.
```

## Current Mainline

| File | Purpose |
|---|---|
| `distributed_sim.py` | Main distributed routing simulator used in current ShareGPT experiments |
| `qmsum_mainline.py` | Clean current QMSum mainline entry: lexical coarse topic routing -> Q-K chunk routing -> answer eval |
| `qmsum_mainline_routing.py` | Clean current QMSum mainline routing core with only lexical coarse routing and topic-local Q-K chunk scoring |
| `qmsum_sim.py` | Legacy/general QMSum experiment entry with older branches and comparisons |
| `qmsum_data.py` | QMSum data loading, prompt building, span/turn conversion, topic-node construction |
| `qmsum_routing.py` | Legacy/general routing logic for older baselines, rerank variants, and broader comparisons |
| `qmsum_answering.py` | Answer generation and answer-F1 helpers |
| `qmsum_output.py` | Case TSV / answer log / markdown output helpers |
| `qmsum_eval.py` | Evaluation and transfer-accounting logic |
| `qmsum_trace.py` | Single-case trace export utilities for step-by-step debugging/explanation |
| `scripts_sharegpt/run_chunk_routing_sweep.sh` | One-click sweep for chunk-to-node aggregation modes with compact summaries |
| `scripts_qmsum/run_qmsum_top1_chunk_budget_sweep.sh` | Focused mainline sweep that fixes top-level routing at 1 topic and only varies chunk budget |
| `scripts_qmsum/run_qmsum_mainline_answer_eval.sh` | One-click answer evaluation for the current frozen QMSum mainline |
| `scripts_qmsum/run_qmsum_neighbor_expand_compare.sh` | Fine-stage comparison for neighbor chunk expansion on top of the frozen mainline |
| `scripts_qmsum/run_qmsum_topic_strategy_compare.sh` | One-click comparison of `embedding` vs `rerank` topic routing under the current mainline setting |
| `scripts_qmsum/run_qmsum_topic_label_ablation.sh` | Topic-label ablation for the current QMSum mainline |
| `scripts_qmsum/run_qmsum_coarse_strategy_explore.sh` | One-click comparison of multiple cheap coarse topic routers: `embedding`, `lexical`, `lexical_prf`, `lexical_hybrid`, `rrf` |
| `experiment_chunk_split.py` | Chunk-level Q-K scoring utilities; provides `_chunk_qk_scores_per_head` used by `distributed_sim.py` |
| `src/utils.py` | Model loading, KV utilities, quantization, and evaluation helpers |
| `data/` | Current local datasets used by the prototype |
| `docs/data_provenance.md` | Dataset source notes and current source-of-truth file mapping |
| `logs/` | Saved console logs from cloud/local experiment runs |
| `outputs/` | JSON outputs written by `distributed_sim.py` |
| `docs/cloud_environment.md` | Cloud runtime environment used for experiments |
| `docs/core_idea.md` | Concise research idea and interpretation |
| `docs/progress.md` | Current status and next-step checklist |

## Repository Layout

```text
root algorithms / simulators
  qmsum_mainline.py
  qmsum_mainline_routing.py
  qmsum_sim.py
  qmsum_data.py
  qmsum_routing.py
  qmsum_answering.py
  qmsum_output.py
  qmsum_eval.py
  qmsum_trace.py
  distributed_sim.py
  experiment_chunk_split.py
  prepare_qmsum_data.py

scripts_qmsum/
  active QMSum experiment scripts

scripts_sharegpt/
  older ShareGPT / flat-routing experiment scripts

docs/
  progress notes, environment notes, paper/code summaries, dataset provenance

_archived_unused_*/
  older branches and no-longer-active experiments
```

If you only want to understand the current working path, focus on:

```text
docs/current_mainline_cn.md
qmsum_mainline_config.py
qmsum_mainline.py
qmsum_mainline_routing.py
qmsum_eval.py
qmsum_data.py
scripts_qmsum/run_qmsum_current_mainline.sh
scripts_qmsum/run_qmsum_mainline_answer_eval.sh
docs/code_reading_map_cn.md
```

The clean current runner is:

```bash
bash scripts_qmsum/run_qmsum_current_mainline.sh
```

It calls:

```text
python qmsum_mainline.py --mainline_profile current
```

Older scripts are still useful for ablations, but the current profile is the
preferred entry point for reading and continuing the mainline.

## Current Best Baseline

As of June 16, 2026, the best validated 25-case QMSum mainline is:

```text
lexical coarse topic routing
-> top-1 topic
-> topic-local Q-K chunk routing
-> dynamic chunk budget
-> qk_then_time answer evidence order
-> selected turns
-> strict answer prompt
```

Observed answer result on docs 5:10, max_queries_per_doc=5:

```text
avg selective-answer F1 = 23.1%
avg answer F1 delta     = +4.2%
selective >= full       = 72.0%
ctx saving              = 86.0%
```

Answer-side variants that were tested but not kept:

```text
chunk_turns + strict
answer_aware rerank + turns + strict
```

So the current heuristic-tuning phase is considered largely stabilized.
The next phase should focus on:

```text
1. more explicit virtual-node distributed simulation for QMSum
2. or two-stage note-compression answer generation
```

## Current Risk Note

After the first system-cost audit on June 17, 2026, the current status should
be read carefully:

```text
selective fetch itself looks promising,
but the online routing implementation is still the dominant latency bottleneck
in the current research prototype.
```

So the next mainline question is no longer only:

```text
can we select useful chunks?
```

It is also:

```text
can routing be made cheap enough, or partially offline enough,
that selective fetch gives a real system-level benefit?
```

## Chinese Quick Map

For the current active QMSum line, the code is now split by responsibility:

```text
qmsum_mainline.py
  main controller
  -> load one meeting / one query
  -> call routing
  -> call evaluation
  -> save summary / trace
  -> save per-case TSV and answer logs

qmsum_data.py
  data preprocessing helpers
  -> load sample
  -> flatten transcript
  -> convert relevant spans to turn ids
  -> build topic nodes

qmsum_mainline_routing.py
  routing core
  -> build topic prototypes
  -> lexical coarse topic scoring
  -> build chunk candidates
  -> select top1 topic
  -> run Q-K chunk routing only inside the chosen topic

qmsum_eval.py
  evaluation core
  -> selected-turn recall / precision / F1
  -> transfer segments / coalescing gain
  -> final console summary

qmsum_trace.py
  explanation / debugging helper
  -> export one-case markdown trace
```

If you want the most readable entry point, start here:

```text
docs/code_reading_map_cn.md
```

## Archived Files

Files from older topic-split / ablation branches were moved to:

```text
_archived_unused_20260605/
```

They are kept for reference, not deleted, but they are not part of the current mainline.

Some lower-priority helper scripts from the earlier flat-routing stage are now
kept under:

```text
_archived_unused_20260606/
```

Some lower-priority QMSum scripts from older precomputed-topic or earlier
comparison branches are now kept under:

```text
_archived_unused_20260619/scripts_qmsum/
```

## Current Experiment Focus

The project now has two branches of work:

```text
Branch A: ShareGPT flat routing
  conversation
  -> split turns across virtual nodes
  -> compute Q-K relevance at chunk level
  -> select top-k chunks
  -> aggregate chunk evidence back to node ranking

Branch B: QMSum hierarchical routing
  meeting transcripts
  -> semantic topic nodes from QMSum topic_list
  -> choose top topic nodes
  -> Q-K chunk routing inside chosen topics
  -> compare selected topics / selected turns against relevant spans
```

The current mainline emphasis is shifting toward Branch B.

Current interpretation:

```text
Flat routing was useful for debugging the routing signal.
But for a realistic distributed setting, nodes should have semantic meaning.
QMSum now serves as the main prototype for this idea.
```

## Local Cleanup Guide

If you want to keep the local checkout easy to navigate, it is useful to split
files into three tiers rather than treating everything in `topic_selective_kv/`
as equally important.

### Tier 1: keep at the top of your attention

These are the current active QMSum mainline files:

```text
qmsum_mainline.py
qmsum_mainline_routing.py
qmsum_data.py
qmsum_eval.py
qmsum_answering.py
qmsum_output.py
qmsum_trace.py
scripts_qmsum/run_qmsum_mainline_answer_eval.sh
scripts_qmsum/run_qmsum_detail_budget_sweep.sh
scripts_qmsum/run_qmsum_prefilter_budget_sweep.sh
scripts_qmsum/run_qmsum_prefilter_budget_sweep_dual_gpu.sh
scripts_qmsum/run_qmsum_qk_batchsize_compare_dual_gpu.sh
docs/progress.md
docs/core_idea.md
docs/code_reading_map_cn.md
```

### Tier 2: keep, but they are not the first place to look

These are still useful support files or side branches, but they are not the
main thing to read every day:

```text
distributed_sim.py
experiment_chunk_split.py
prepare_qmsum_data.py
qmsum_sim.py
qmsum_routing.py
scripts_sharegpt/
scripts_qmsum/run_qmsum_neighbor_expand_compare.sh
scripts_qmsum/run_qmsum_top1_chunk_budget_sweep.sh
scripts_qmsum/run_qmsum_coarse_strategy_explore.sh
scripts_qmsum/run_qmsum_topic_strategy_compare.sh
scripts_qmsum/run_qmsum_topic_label_ablation.sh
scripts_qmsum/run_qmsum_answer_ablation_dual_gpu.sh
scripts_qmsum/preview_qmsum_sample.py
```

Interpretation:

```text
important enough to keep
but not the current narrow mainline
```

### Tier 3: archived on 2026-06-19

These are the least important local scripts for the current direction because
they mostly belong to older precomputed-topic or earlier comparison branches:

```text
_archived_unused_20260619/scripts_qmsum/run_qmsum_precomputed_rerank_compare.sh
_archived_unused_20260619/scripts_qmsum/run_qmsum_precomputed_topic_sweep.sh
_archived_unused_20260619/scripts_qmsum/run_qmsum_topic_embedding_source_compare.sh
_archived_unused_20260619/scripts_qmsum/run_qmsum_topic_repr_template_compare.sh
_archived_unused_20260619/scripts_qmsum/run_qmsum_hierarchical_sweep.sh
_archived_unused_20260619/scripts_qmsum/run_qmsum_mainline_answer_eval_20docs_all_queries.sh
_archived_unused_20260619/scripts_qmsum/run_qmsum_mainline_answer_eval_20docs_dual_gpu.sh
```

These are not "wrong" files.
They are simply not the most relevant scripts for the current local workflow:

```text
lexical coarse topic routing
-> topic-local Q-K
-> detail-query answer quality
-> communication / routing-overhead reduction
```

### Safe cleanup rule

For now, the safest cleanup is:

```text
1. do not delete Tier 2 or Tier 3 files yet
2. keep Tier 1 visible and mentally primary
3. Tier 3 has already been physically archived
4. only archive Tier 2 after the current QMSum mainline is stable enough
```

Latest routing update:

```text
The current bottleneck is coarse topic routing.
Precomputed embedding alone can miss the correct top-1 topic on hard nearby-topic cases.
So the active coarse-routing comparison now includes:
  embedding
  lexical
  lexical_prf
  lexical_hybrid
  rrf
while Q-K remains the fine-stage selector inside the chosen topic.
```

Additional clarification for the current QMSum branch:

```text
We are not currently claiming that the system can automatically build good topic nodes.
For now, QMSum's labeled topic spans provide the semantic partition.
The research target is the routing policy under that partition.
```

## Current Commands

Run a single chunk-routing experiment:

```bash
python distributed_sim.py --model_id ~/models/mistral-7b/ \
    --num_gpus 1 --max_gpu_memory 40 \
    --num_nodes 4 --start_doc 0 --end_doc 200 \
    --passkey --baselines \
    --routing_granularity chunk \
    --route_chunk_size 128 \
    --route_top_k 4 \
    --route_per_head \
    --chunk_node_score_mode selected_count
```

Run the one-click aggregation sweep:

```bash
bash scripts_sharegpt/run_chunk_routing_sweep.sh
```

This produces:

```text
logs/chunk_node_score_sweep/summary.tsv
logs/chunk_node_score_sweep/summary.txt
```

For QMSum runs, `qmsum_sim.py` now also saves a per-case comparison file:

```text
outputs/qmsum_case_summary_N*_*.tsv
```

This file is useful for checking whether different coarse strategies
really chose different topics on each `(doc, query)` sample.

The summary files keep only the key metrics:

```text
qk top-1
qk top-2
qk selected-node
qk selected-turn
qk selected-first-chunk
Q-K variance / range
distinguishable ratio
```

Run the minimal QMSum routing simulation:

```bash
python qmsum_sim.py --data_path ~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl \
    --model_id ~/models/mistral-7b/ \
    --num_gpus 1 --max_gpu_memory 40 \
    --num_nodes 4 --start_doc 0 --end_doc 5 \
    --max_queries_per_doc 2 \
    --baselines \
    --routing_granularity chunk \
    --route_chunk_size 128 \
    --route_top_k 4 \
    --route_per_head
```

This checks whether selected chunks and selected nodes cover QMSum's native
`relevant_text_span` annotations.

Run the current hierarchical QMSum prototype:

```bash
python qmsum_sim.py --data_path ~/working_place/topic_selective_kv/data/qmsum_structured/train.jsonl \
    --model_id ~/models/mistral-7b/ \
    --num_gpus 1 --max_gpu_memory 40 \
    --start_doc 0 --end_doc 5 \
    --max_queries_per_doc 2 \
    --baselines \
    --routing_granularity hierarchical \
    --hier_top_topics 1 \
    --hier_top_strategy lexical \
    --hier_topic_score_mode sum \
    --route_chunk_size 128 \
    --route_top_k 12 \
    --route_per_head
```

This uses:

```text
lexical topic routing
-> Q-K chunk routing
```

Run the small hierarchical sweep:

```bash
bash _archived_unused_20260619/scripts_qmsum/run_qmsum_hierarchical_sweep.sh
```

This produces:

```text
logs/qmsum_hierarchical_sweep/summary.tsv
logs/qmsum_hierarchical_sweep/summary.txt
```

The summary now includes a `status` column, so failed runs show up as
`RUN_FAIL` or `PARSE_FAIL` instead of a blank row.

Run the focused top1-topic mainline sweep:

```bash
bash scripts_qmsum/run_qmsum_top1_chunk_budget_sweep.sh
```

This produces:

```text
logs/qmsum_top1_chunk_budget_sweep/summary.tsv
logs/qmsum_top1_chunk_budget_sweep/summary.txt
```

Run the current frozen answer-evaluation mainline:

```bash
bash scripts_qmsum/run_qmsum_mainline_answer_eval.sh
```

Run the current fine-stage neighbor expansion compare:

```bash
bash scripts_qmsum/run_qmsum_neighbor_expand_compare.sh
```

Run the current topic-routing accuracy comparison:

```bash
bash scripts_qmsum/run_qmsum_topic_strategy_compare.sh
```

This produces:

```text
logs/qmsum_topic_strategy_compare/summary.tsv
logs/qmsum_topic_strategy_compare/summary.txt
```

Run the topic-label ablation:

```bash
bash scripts_qmsum/run_qmsum_topic_label_ablation.sh
```

This compares:

```text
topic_label_weight=0.0
vs
topic_label_weight=0.35
```

## Current Interpretation

Recent results suggest:

```text
1. Chunk-level Q-K retrieval is much better than node-level scoring for finding relevant evidence.
2. The selected chunk set often covers the correct turn/chunk.
3. Count-based aggregation is currently the best chunk -> node rule on ShareGPT passkey.
4. Embedding is still a very strong baseline on the synthetic passkey task.
```

Additional current QMSum conclusion:

```text
1. Flat QMSum node metrics were initially too loose under round-robin assignment.
2. Contiguous node assignment made the evaluation more realistic.
3. Pure Q-K topic routing performed badly.
4. Lexical coarse routing is currently stronger than precomputed embedding on the validated small slice.
5. The current frozen mainline is:
   lexical top1 topic routing
   + Q-K chunk selection
   + route_top_k=12
6. Direct hybrid score mixing was tried and removed from the main path because it hurt accuracy.
```

Updated mainline takeaway from the topic-label ablation:

```text
When topic labels are allowed to contribute to coarse routing,
top-level topic hit and turn-level evidence quality improve substantially.
This supports the current mainline assumption:
  labeled semantic nodes are useful and acceptable at this stage.
```

## Current Limitation

The current hierarchical prototype should not be oversold as a final efficient router:

```text
Although the logic is "topic first, chunk second",
the implementation still computes many chunk Q-K scores globally before final filtering.
So the current code is best understood as a research prototype for routing quality,
not yet a fully optimized low-overhead serving path.
```

## Next Mainline

The next improvement direction is now narrower and clearer:

```text
1. Keep labeled topics / labeled turn-to-topic mapping
2. Keep lexical as the coarse topic router
3. Keep Q-K as the in-topic fine selector
4. Freeze `route_top_k=12` as the current balanced default
5. Add final answer generation comparison:
   full context vs selective context
6. Compare local neighbor chunk expansion on top of the fixed mainline
7. Add simple fine-stage statistical baselines:
   mean / variance / max / min / top-k
6. Compare both accuracy and transfer reduction
```

## Notes On Local vs Cloud

- Cloud results are the source of truth for experiment outputs.
- Local files are the editable/reference copy.
- Cloud environment details are in `docs/cloud_environment.md`.
