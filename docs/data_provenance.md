# Data Provenance

This note records where the currently used datasets came from, what format they are in now, and what role they play in the current experiments.

## ShareGPT

### Current file used by the simulator

```text
CacheGen-main/test_data/sharegpt.jsonl
```

This is the file currently used by:

```text
distributed_sim.py
```

and therefore by the ShareGPT routing experiments.

### Current status

- Current file size is small and contains about `200` samples.
- It is being used as a lightweight experiment subset for the current routing simulation.

### Likely source chain

The most likely upstream source is:

```text
ShareGPT_V3_unfiltered_cleaned_split.json
```

This is supported by the reference repository:

```text
reference_repos/KVDirect-main/benchmarks/README.md
```

which downloads:

```text
https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Current best reconstruction of the lineage:

```text
KVDirect / ShareGPT upstream file
-> local conversion or sampling
-> CacheGen-main/test_data/sharegpt.jsonl
-> topic_selective_kv/distributed_sim.py
```

### Important note

We did not find a clearly preserved local conversion script for this exact subset in the current workspace.

So the confirmed part is:

```text
The current simulator reads CacheGen-main/test_data/sharegpt.jsonl.
The likely original source is the ShareGPT file referenced in KVDirect.
```

The unconfirmed part is:

```text
Exactly which script produced this 200-sample JSONL subset.
```

## QMSum

### Structured source

Official raw source used for preparation:

```text
reference_repos/QMSum-main/data/ALL/jsonl/{train,val,test}.jsonl
```

### Prepared local format

Prepared by:

```text
prepare_qmsum_data.py
```

into:

```text
topic_selective_kv/data/qmsum_structured/{train,val,test}.jsonl
```

### Why preparation is needed

The flattened files under:

```text
topic_selective_kv/data/qmsum/{train,val,test}.txt
```

are prompt-style text files and do not preserve the structured fields needed for routing experiments, such as:

- `meeting_transcripts`
- `specific_query_list`
- `relevant_text_span`

So the structured JSONL source is the one we should treat as the real routing input source.

## LongChat

Current local file:

```text
topic_selective_kv/data/longchat.jsonl
```

This dataset mainly belongs to the earlier chunk-selection line rather than the current ShareGPT distributed routing mainline.

## Practical Rule

For the current project, use the following source-of-truth mapping:

```text
ShareGPT routing:
  CacheGen-main/test_data/sharegpt.jsonl

QMSum routing:
  topic_selective_kv/data/qmsum_structured/*.jsonl

Raw QMSum source:
  reference_repos/QMSum-main/data/ALL/jsonl/*.jsonl
```

