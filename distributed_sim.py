"""
Distributed KV routing simulation on ShareGPT conversations.

Two routing modes are supported:

1. node:
   Collapse each virtual node into one Q-K score. This is the old behavior.

2. chunk:
   Split each turn into smaller token chunks, score every chunk globally, select
   top-k chunks, and map selected chunks back to nodes. This is closer to Quest
   and is more suitable for needle/passkey tests.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_sharegpt_sample(filepath, doc_id):
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == doc_id:
                return json.loads(line.strip())
    raise IndexError(f"doc_id={doc_id} out of range")


def build_conversation_prompt(conversations, tokenizer):
    parts = []
    turn_boundaries = []
    offset = 0

    for turn in conversations:
        role = turn["user"]
        prefix = "USER: " if role == "human" else "ASSISTANT: "
        turn_text = prefix + turn["text"] + "\n\n"
        turn_ids = tokenizer(turn_text, add_special_tokens=False).input_ids
        n_tok = len(turn_ids)

        parts.append(turn_text)
        turn_boundaries.append((offset, offset + n_tok))
        offset += n_tok

    return "".join(parts), turn_boundaries


def build_query_from_last_user_message(conversations):
    for turn in reversed(conversations):
        if turn["user"] == "human":
            return turn["text"].strip()
    return conversations[-1]["text"].strip()


_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            model_name = os.environ.get(
                "TOPIC_ROUTING_EMBED_MODEL",
                "all-MiniLM-L6-v2",
            )
            local_only_env = os.environ.get("TOPIC_ROUTING_EMBED_LOCAL_ONLY", "1").strip().lower()
            local_only = local_only_env not in {"0", "false", "no"}

            try:
                _embedding_model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    local_files_only=local_only,
                )
            except TypeError:
                if local_only:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                _embedding_model = SentenceTransformer(model_name, device="cpu")
            except Exception as exc:
                if local_only:
                    print(
                        "  [WARN] failed to load embedding model from local cache only; "
                        "set TOPIC_ROUTING_EMBED_LOCAL_ONLY=0 if you want to allow network download"
                    )
                raise exc
        except ImportError:
            print("  [WARN] sentence-transformers is not installed; skip embedding baseline")
            _embedding_model = False
    return _embedding_model if _embedding_model is not False else None


def score_nodes_random(num_nodes, seed=None):
    rng = np.random.RandomState(seed)
    return {n: float(rng.random()) for n in range(num_nodes)}


def score_nodes_recency(node_turn_info, num_nodes):
    scores = {}
    for n in range(num_nodes):
        turns = node_turn_info.get(n, [])
        scores[n] = float(max(turns)) if turns else -1e9
    return scores


def score_nodes_embedding(query_text, node_turn_info, conversations):
    model = _get_embedding_model()
    if model is None:
        return {n: -1e9 for n in node_turn_info}

    query_emb = model.encode([query_text])[0]

    turn_texts = []
    for turn in conversations:
        prefix = "USER: " if turn["user"] == "human" else "ASSISTANT: "
        turn_texts.append(prefix + turn["text"] + "\n\n")

    scores = {}
    for node_id, turns in node_turn_info.items():
        if not turns:
            scores[node_id] = -1e9
            continue

        sims = []
        for turn_idx in turns:
            if turn_idx >= len(turn_texts):
                continue
            text = turn_texts[turn_idx].strip()
            if not text:
                continue
            text_emb = model.encode([text])[0]
            sim = float(
                np.dot(query_emb, text_emb)
                / (np.linalg.norm(query_emb) * np.linalg.norm(text_emb) + 1e-8)
            )
            sims.append(sim)
        scores[node_id] = float(np.mean(sims)) if sims else -1e9
    return scores


def aggregate_qk_scores(scores_matrix, mode="mean", topk=4):
    flat = np.asarray(scores_matrix, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return -1e9
    if mode == "mean":
        return float(flat.mean())
    if mode == "max":
        return float(flat.max())
    if mode == "topk_mean":
        k = max(1, min(int(topk), flat.size))
        return float(np.partition(flat, -k)[-k:].mean())
    raise ValueError(f"Unknown qk aggregation mode: {mode}")


def rank_scores(scores):
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return {
        "top_node": int(ranked[0][0]),
        "ranked_nodes": [int(n) for n, _ in ranked],
        "ranked_scores": [float(s) for _, s in ranked],
    }


def stack_keys_from_kv(kv_chunk):
    return torch.stack([layer[0].squeeze(0) for layer in kv_chunk], dim=0)


def assign_turn_to_node(turn_idx, num_turns, num_nodes, mode="round_robin"):
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if mode == "round_robin":
        return int(turn_idx % num_nodes)
    if mode == "contiguous":
        block_size = max(1, int(np.ceil(num_turns / max(1, num_nodes))))
        return int(min(turn_idx // block_size, num_nodes - 1))
    raise ValueError(f"Unknown node assignment mode: {mode}")


def build_node_turn_info(turn_boundaries, num_nodes, node_assignment_mode):
    node_turn_info = {n: [] for n in range(num_nodes)}
    num_turns = len(turn_boundaries)
    for turn_idx, (start_t, end_t) in enumerate(turn_boundaries):
        if end_t <= start_t:
            continue
        node_id = assign_turn_to_node(
            turn_idx, num_turns, num_nodes, mode=node_assignment_mode
        )
        node_turn_info[node_id].append(turn_idx)
    return node_turn_info


def build_route_candidates(
    turn_boundaries, total_tokens, num_nodes, chunk_size, node_assignment_mode
):
    candidates = []
    node_turn_info = {n: [] for n in range(num_nodes)}
    num_turns = len(turn_boundaries)

    for turn_idx, (start_t, end_t) in enumerate(turn_boundaries):
        if end_t <= start_t or start_t >= total_tokens:
            continue

        node_id = assign_turn_to_node(
            turn_idx, num_turns, num_nodes, mode=node_assignment_mode
        )
        node_turn_info[node_id].append(turn_idx)

        local_chunk_idx = 0
        for c_start in range(start_t, min(end_t, total_tokens), chunk_size):
            c_end = min(c_start + chunk_size, end_t, total_tokens)
            if c_end <= c_start:
                continue
            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "node_id": int(node_id),
                    "turn_idx": int(turn_idx),
                    "local_chunk_idx": int(local_chunk_idx),
                    "start_t": int(c_start),
                    "end_t": int(c_end),
                    "n_tokens": int(c_end - c_start),
                }
            )
            local_chunk_idx += 1

    return candidates, node_turn_info


def score_node_level(
    kv_3d,
    turn_boundaries,
    query_ids,
    model,
    scoring_layers,
    split_kv,
    qk_score_fn,
    num_nodes,
    args,
):
    node_scores = {n: -1e9 for n in range(num_nodes)}
    node_turn_info = build_node_turn_info(
        turn_boundaries, num_nodes, args.node_assignment_mode
    )

    for node_id in range(num_nodes):
        turn_scores = []
        for turn_idx in node_turn_info[node_id]:
            start_t, end_t = turn_boundaries[turn_idx]
            kv_chunk = split_kv(kv_3d, start_t, end_t)
            k_stacked = stack_keys_from_kv(kv_chunk)
            scores, _ = qk_score_fn(query_ids, [k_stacked], model, scoring_layers)
            turn_scores.append(
                aggregate_qk_scores(scores, args.qk_aggregation, args.qk_topk)
            )
            del kv_chunk, k_stacked, scores

        if turn_scores:
            node_scores[node_id] = aggregate_qk_scores(
                turn_scores, args.qk_aggregation, args.qk_topk
            )

    return node_scores, node_turn_info, {}


def score_chunk_level(
    kv_3d,
    turn_boundaries,
    total_tokens,
    query_ids,
    model,
    scoring_layers,
    split_kv,
    qk_score_fn,
    num_nodes,
    args,
):
    route_candidates, node_turn_info = build_route_candidates(
        turn_boundaries,
        total_tokens,
        num_nodes,
        max(1, args.route_chunk_size),
        args.node_assignment_mode,
    )
    legacy_node_scores = {n: -1e9 for n in range(num_nodes)}
    scored_candidates = []
    per_head_scores = []

    for cand in route_candidates:
        kv_chunk = split_kv(kv_3d, cand["start_t"], cand["end_t"])
        k_stacked = stack_keys_from_kv(kv_chunk)
        scores, _ = qk_score_fn(query_ids, [k_stacked], model, scoring_layers)
        scalar = aggregate_qk_scores(scores, args.qk_aggregation, args.qk_topk)

        record = dict(cand)
        record["score"] = float(scalar)
        scored_candidates.append(record)
        per_head_scores.append(np.asarray(scores, dtype=np.float32).reshape(-1))
        legacy_node_scores[record["node_id"]] = max(
            legacy_node_scores[record["node_id"]], scalar
        )

        del kv_chunk, k_stacked, scores

    selected_candidates = []
    if scored_candidates:
        if args.route_per_head and per_head_scores:
            score_matrix = np.stack(per_head_scores, axis=0)
            selected_ids = set()
            k = min(max(1, args.route_top_k), len(scored_candidates))
            for head_idx in range(score_matrix.shape[1]):
                top_local = np.argsort(score_matrix[:, head_idx])[-k:]
                selected_ids.update(int(i) for i in top_local)
            selected_candidates = [scored_candidates[i] for i in selected_ids]
            selected_candidates = sorted(selected_candidates, key=lambda x: -x["score"])
        else:
            k = min(max(1, args.route_top_k), len(scored_candidates))
            selected_candidates = sorted(
                scored_candidates, key=lambda x: -x["score"]
            )[:k]

    selected_nodes = []
    seen_nodes = set()
    for cand in selected_candidates:
        node_id = cand["node_id"]
        if node_id not in seen_nodes:
            selected_nodes.append(int(node_id))
            seen_nodes.add(node_id)

    node_scores = {n: -1e9 for n in range(num_nodes)}
    if args.chunk_node_score_mode == "all_chunk_max":
        node_scores = dict(legacy_node_scores)
    else:
        node_score_buckets = defaultdict(list)
        for cand in selected_candidates:
            node_score_buckets[cand["node_id"]].append(float(cand["score"]))

        for node_id, bucket in node_score_buckets.items():
            if args.chunk_node_score_mode == "selected_sum":
                node_scores[node_id] = float(sum(bucket))
            elif args.chunk_node_score_mode == "selected_count":
                node_scores[node_id] = float(len(bucket))
            elif args.chunk_node_score_mode == "selected_max":
                node_scores[node_id] = float(max(bucket))
            else:
                raise ValueError(
                    f"Unknown chunk node score mode: {args.chunk_node_score_mode}"
                )

    selected_node_scores = {
        int(node_id): float(node_scores[node_id])
        for node_id in selected_nodes
        if node_scores[node_id] > -1e8
    }

    route_info = {
        "granularity": "chunk",
        "route_chunk_size": args.route_chunk_size,
        "route_top_k": args.route_top_k,
        "route_per_head": args.route_per_head,
        "chunk_node_score_mode": args.chunk_node_score_mode,
        "num_candidates": len(scored_candidates),
        "num_selected_candidates": len(selected_candidates),
        "selected_nodes": selected_nodes,
        "selected_node_scores": selected_node_scores,
        "selected_token_count": int(sum(c["n_tokens"] for c in selected_candidates)),
        "selected_candidates": [
            {
                "node_id": c["node_id"],
                "turn_idx": c["turn_idx"],
                "local_chunk_idx": c["local_chunk_idx"],
                "start_t": c["start_t"],
                "end_t": c["end_t"],
                "n_tokens": c["n_tokens"],
                "score": c["score"],
            }
            for c in selected_candidates[:20]
        ],
    }

    return node_scores, node_turn_info, route_info


def main(args):
    print("=" * 70)
    print("Distributed KV routing simulation (ShareGPT)")
    print(f"  nodes: {args.num_nodes}")
    print(f"  data: {args.data_path}")
    print(f"  node_assignment_mode: {args.node_assignment_mode}")
    print(f"  routing_granularity: {args.routing_granularity}")
    if args.routing_granularity == "chunk":
        print(f"  chunk_node_score_mode: {args.chunk_node_score_mode}")
    print("=" * 70)

    from src.utils import define_model_and_tokenizer, split_kv
    from experiment_chunk_split import _chunk_qk_scores_per_head

    print("\nLoading model...")
    model, tokenizer = define_model_and_tokenizer(
        args.model_id,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
    )
    print("Model loaded\n")

    num_model_layers = len(model.model.layers)
    if args.scoring_layers:
        scoring_layers = [int(x) for x in args.scoring_layers.split(",")]
    else:
        scoring_layers = [0, 8, 16, 24, num_model_layers - 1]
    scoring_layers = [l if l >= 0 else num_model_layers + l for l in scoring_layers]

    num_nodes = args.num_nodes
    results = []

    for doc_id in range(args.start_doc, args.end_doc):
        try:
            data = load_sharegpt_sample(args.data_path, doc_id=doc_id)
        except IndexError:
            break

        conversations = data["conversations"]
        n_turns = len(conversations)
        if n_turns < 4:
            continue

        passkey_info = {}
        if args.passkey:
            rng = np.random.RandomState(doc_id)
            passkey_value = int(rng.randint(10000, 99999))
            passkey_node = int(rng.randint(0, num_nodes))
            candidate_turns = [
                ti
                for ti in range(n_turns)
                if assign_turn_to_node(
                    ti, n_turns, num_nodes, args.node_assignment_mode
                )
                == passkey_node
            ]
            if candidate_turns:
                passkey_turn = int(candidate_turns[rng.randint(0, len(candidate_turns))])
                old_text = conversations[passkey_turn]["text"]
                conversations[passkey_turn]["text"] = (
                    f"The secret pass key is {passkey_value}. Remember it. "
                    f"{passkey_value} is the pass key. {old_text}"
                )
                passkey_info = {
                    "passkey_node": passkey_node,
                    "passkey_value": passkey_value,
                    "passkey_turn": passkey_turn,
                }

        prompt_text, turn_boundaries = build_conversation_prompt(conversations, tokenizer)
        if args.passkey and passkey_info:
            query_text = "What is the pass key? The pass key is"
        else:
            query_text = build_query_from_last_user_message(conversations)

        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        total_tokens = input_ids.shape[1]

        if total_tokens > args.max_tokens:
            print(
                f"  [SKIP] doc_id={doc_id}: {total_tokens} tokens > {args.max_tokens}"
            )
            del input_ids
            torch.cuda.empty_cache()
            continue

        try:
            with torch.no_grad():
                outputs = model.model(input_ids=input_ids, use_cache=True)
            kv_3d = tuple((layer[0][0], layer[1][0]) for layer in outputs.past_key_values)
            kv_4d = outputs.past_key_values
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM SKIP] doc_id={doc_id}: {total_tokens} tokens")
            del input_ids
            torch.cuda.empty_cache()
            continue

        query_ids = tokenizer(query_text, return_tensors="pt").input_ids.cuda()

        if args.routing_granularity == "node":
            node_scores, node_turn_info, qk_route_info = score_node_level(
                kv_3d,
                turn_boundaries,
                query_ids,
                model,
                scoring_layers,
                split_kv,
                _chunk_qk_scores_per_head,
                num_nodes,
                args,
            )
        else:
            node_scores, node_turn_info, qk_route_info = score_chunk_level(
                kv_3d,
                turn_boundaries,
                total_tokens,
                query_ids,
                model,
                scoring_layers,
                split_kv,
                _chunk_qk_scores_per_head,
                num_nodes,
                args,
            )

        qk_values = [s for s in node_scores.values() if s > -1e8]
        score_variance = float(np.var(qk_values)) if len(qk_values) > 1 else 0.0
        score_range = float(np.max(qk_values) - np.min(qk_values)) if qk_values else 0.0

        all_strategies = {"qk": node_scores}
        if args.baselines:
            all_strategies["random"] = score_nodes_random(num_nodes, seed=doc_id)
            all_strategies["recency"] = score_nodes_recency(node_turn_info, num_nodes)
            all_strategies["embedding"] = score_nodes_embedding(
                query_text, node_turn_info, conversations
            )

        strategy_results = {
            strat_name: rank_scores(scores)
            for strat_name, scores in all_strategies.items()
        }
        if qk_route_info:
            strategy_results["qk"]["route_info"] = qk_route_info

        top_node = strategy_results["qk"]["top_node"]
        top_turns = node_turn_info.get(top_node, [])

        result_entry = {
            "doc_id": doc_id,
            "n_turns": n_turns,
            "total_tokens": int(total_tokens),
            "num_nodes": num_nodes,
            "qk_score_variance": score_variance,
            "qk_score_range": score_range,
            "top_node": top_node,
            "top_node_turns": top_turns,
            "query_preview": query_text[:100],
            "strategy_results": strategy_results,
        }
        if qk_route_info:
            result_entry["qk_route_info"] = qk_route_info

        if args.passkey and passkey_info:
            strategy_hits = {}
            for strat_name, sr in strategy_results.items():
                strategy_hits[strat_name] = {
                    "top1_hit": sr["top_node"] == passkey_info["passkey_node"],
                    "top2_hit": passkey_info["passkey_node"] in sr["ranked_nodes"][:2],
                }

            selected_nodes = qk_route_info.get("selected_nodes", []) if qk_route_info else []
            selected_candidates = (
                qk_route_info.get("selected_candidates", []) if qk_route_info else []
            )
            selected_node_hit = passkey_info["passkey_node"] in selected_nodes
            selected_turn_hit = any(
                c["turn_idx"] == passkey_info["passkey_turn"]
                for c in selected_candidates
            )
            selected_first_chunk_hit = any(
                c["turn_idx"] == passkey_info["passkey_turn"]
                and c["local_chunk_idx"] == 0
                for c in selected_candidates
            )

            result_entry["passkey"] = {
                "node": passkey_info["passkey_node"],
                "value": passkey_info["passkey_value"],
                "turn": passkey_info["passkey_turn"],
                "qk_top1_hit": strategy_hits["qk"]["top1_hit"],
                "qk_top2_hit": strategy_hits["qk"]["top2_hit"],
                "qk_selected_node_hit": selected_node_hit,
                "qk_selected_turn_hit": selected_turn_hit,
                "qk_selected_first_chunk_hit": selected_first_chunk_hit,
                "strategy_hits": strategy_hits,
            }

        results.append(result_entry)

        if doc_id < args.start_doc + 3:
            print("\n" + "-" * 60)
            print(f"[{doc_id}] {n_turns} turns, {total_tokens} tokens, {num_nodes} nodes")
            print(f"  Query: {query_text[:100]!r}")
            print(f"  Q-K node scores: {dict((f'N{k}', f'{v:.4f}') for k, v in sorted(node_scores.items()))}")
            print(f"  Q-K variance: {score_variance:.6f}, range: {score_range:.4f}")
            print(f"  Q-K top node: N{top_node} (turns {top_turns})")
            if qk_route_info:
                print(
                    f"  selected chunks: {qk_route_info['num_selected_candidates']}/"
                    f"{qk_route_info['num_candidates']}, nodes={qk_route_info['selected_nodes']}"
                )
            if args.baselines:
                for sname in ["random", "recency", "embedding"]:
                    sr = strategy_results[sname]
                    print(
                        f"  {sname:>9}: Top=N{sr['top_node']}, "
                        f"ranked={sr['ranked_nodes'][:4]}"
                    )

        del input_ids, kv_3d, kv_4d, query_ids
        torch.cuda.empty_cache()

    n = len(results)
    if n == 0:
        print("No results")
        return

    avg_variance = np.mean([r["qk_score_variance"] for r in results])
    avg_range = np.mean([r["qk_score_range"] for r in results])
    high_var = sum(1 for r in results if r["qk_score_variance"] > 0.001)
    avg_turns = np.mean([r["n_turns"] for r in results])
    strategy_keys = list(results[0]["strategy_results"].keys())

    def get_top_node_dist(strategy_name):
        dist = defaultdict(int)
        for r in results:
            dist[r["strategy_results"][strategy_name]["top_node"]] += 1
        return dict(sorted(dist.items()))

    print(f"\n{'=' * 70}")
    print(f"Distributed KV routing summary (ShareGPT, n={n})")
    print(f"{'=' * 70}")
    print(f"  avg turns:              {avg_turns:.1f}")
    print(f"  nodes:                  {num_nodes}")
    print(f"  Q-K avg score variance: {avg_variance:.6f}")
    print(f"  Q-K avg score range:    {avg_range:.4f}")
    print(f"  distinguishable:        {high_var}/{n} ({100 * high_var / n:.0f}%)")
    print("\n  Top-node distribution:")
    for sname in sorted(strategy_keys):
        dist = get_top_node_dist(sname)
        dist_str = ", ".join(f"N{k}:{v}" for k, v in dist.items())
        print(f"    {sname:>9}: {dist_str} (hit {len(dist)}/{num_nodes} nodes)")

    passkey_summary = {}
    if args.passkey:
        passkey_results = [r["passkey"] for r in results if "passkey" in r]
        n_pk = len(passkey_results)
        if n_pk > 0:
            per_strategy = {}
            for sname in strategy_keys:
                top1_hits = sum(
                    1
                    for p in passkey_results
                    if p.get("strategy_hits", {}).get(sname, {}).get("top1_hit", False)
                )
                top2_hits = sum(
                    1
                    for p in passkey_results
                    if p.get("strategy_hits", {}).get(sname, {}).get("top2_hit", False)
                )
                per_strategy[sname] = {
                    "top1_hits": top1_hits,
                    "top2_hits": top2_hits,
                    "top1_hit_rate": top1_hits / n_pk,
                    "top2_hit_rate": top2_hits / n_pk,
                }

            selected_node_hits = sum(
                1 for p in passkey_results if p.get("qk_selected_node_hit", False)
            )
            selected_turn_hits = sum(
                1 for p in passkey_results if p.get("qk_selected_turn_hit", False)
            )
            selected_first_chunk_hits = sum(
                1
                for p in passkey_results
                if p.get("qk_selected_first_chunk_hit", False)
            )

            qk_hits = per_strategy["qk"]
            passkey_summary = {
                "top1_hit_rate": qk_hits["top1_hit_rate"],
                "top2_hit_rate": qk_hits["top2_hit_rate"],
                "qk_selected_node_hit_rate": selected_node_hits / n_pk,
                "qk_selected_turn_hit_rate": selected_turn_hits / n_pk,
                "qk_selected_first_chunk_hit_rate": selected_first_chunk_hits / n_pk,
                "per_strategy": per_strategy,
            }

            print("\n  Passkey routing hit rate:")
            for sname in sorted(per_strategy.keys()):
                sm = per_strategy[sname]
                print(
                    f"    {sname:>9} top-1: {sm['top1_hits']}/{n_pk} "
                    f"({100 * sm['top1_hit_rate']:.1f}%), "
                    f"top-2: {sm['top2_hits']}/{n_pk} "
                    f"({100 * sm['top2_hit_rate']:.1f}%)"
                )
            if args.routing_granularity != "node":
                print(
                    f"    qk selected-node: {selected_node_hits}/{n_pk} "
                    f"({100 * selected_node_hits / n_pk:.1f}%)"
                )
                print(
                    f"    qk selected-turn: {selected_turn_hits}/{n_pk} "
                    f"({100 * selected_turn_hits / n_pk:.1f}%)"
                )
                print(
                    f"    qk selected-first-chunk: {selected_first_chunk_hits}/{n_pk} "
                    f"({100 * selected_first_chunk_hits / n_pk:.1f}%)"
                )
        else:
            passkey_summary = {"top1_hit_rate": None, "top2_hit_rate": None}

    per_strategy_summary = {}
    for sname in strategy_keys:
        dist = get_top_node_dist(sname)
        per_strategy_summary[sname] = {
            "top_node_distribution": {str(k): v for k, v in dist.items()},
            "unique_nodes_hit": len(dist),
        }

    out_path = f"outputs/dist_sim_sharegpt_N{num_nodes}_{args.start_doc}_{args.end_doc}.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "num_nodes": num_nodes,
                    "dataset": "sharegpt",
                    "scoring_layers": args.scoring_layers,
                    "routing_granularity": args.routing_granularity,
                    "route_chunk_size": args.route_chunk_size,
                    "route_top_k": args.route_top_k,
                    "route_per_head": args.route_per_head,
                    "node_assignment_mode": args.node_assignment_mode,
                    "chunk_node_score_mode": args.chunk_node_score_mode,
                    "qk_aggregation": args.qk_aggregation,
                    "qk_topk": args.qk_topk,
                    "baselines": args.baselines,
                    "passkey": args.passkey,
                },
                "summary": {
                    "n": n,
                    "avg_turns": float(avg_turns),
                    "qk_avg_score_variance": float(avg_variance),
                    "qk_avg_score_range": float(avg_range),
                    "qk_high_variance_ratio": high_var / n if n > 0 else 0,
                    "per_strategy": per_strategy_summary,
                    "passkey": passkey_summary,
                },
                "details": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Distributed KV routing simulation")
    p.add_argument("--dataset", type=str, default="sharegpt")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_gpu_memory", type=int, default=48)
    p.add_argument("--num_nodes", type=int, default=4)
    p.add_argument("--scoring_layers", type=str, default=None)
    p.add_argument("--start_doc", type=int, default=0)
    p.add_argument("--end_doc", type=int, default=10)
    p.add_argument("--baselines", action="store_true", default=False)
    p.add_argument("--max_tokens", type=int, default=15000)
    p.add_argument("--passkey", action="store_true", default=False)
    p.add_argument("--routing_granularity", type=str, default="node", choices=["node", "chunk"])
    p.add_argument(
        "--node_assignment_mode",
        type=str,
        default="round_robin",
        choices=["round_robin", "contiguous"],
    )
    p.add_argument("--route_chunk_size", type=int, default=128)
    p.add_argument("--route_top_k", type=int, default=8)
    p.add_argument("--route_per_head", action="store_true", default=False)
    p.add_argument(
        "--chunk_node_score_mode",
        type=str,
        default="selected_count",
        choices=["selected_sum", "selected_count", "selected_max", "all_chunk_max"],
    )
    p.add_argument("--qk_aggregation", type=str, default="mean", choices=["mean", "max", "topk_mean"])
    p.add_argument("--qk_topk", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if args.data_path is None:
        base = os.path.join(os.path.dirname(__file__), "..", "CacheGen-main", "test_data")
        args.data_path = os.path.abspath(os.path.join(base, "sharegpt.jsonl"))

    if not os.path.exists(args.data_path):
        print(f"Data file does not exist: {args.data_path}")
        sys.exit(1)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    main(args)
