"""
Legacy/general QMSum routing simulation entry.

This file keeps historical experiment branches together:
node/chunk/hierarchical routing, baseline comparisons, rerank variants, and
older ablations. The cleaner current mainline entry is qmsum_mainline.py.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qmsum_data import (
    build_qmsum_answer_prompt,
    build_qmsum_prompt,
    load_qmsum_sample,
    spans_to_turn_set,
)
from qmsum_eval import (
    build_summary_payload,
    build_transfer_accounting,
    compute_selected_turn_metrics,
    print_summary,
)
from qmsum_routing import (
    build_topic_embedding_index_qmsum,
    score_hierarchical_topic_chunk,
    score_nodes_embedding_qmsum,
)
from qmsum_trace import preview_text, should_trace_case, write_case_trace


def generate_answer_text(model, tokenizer, prompt_text, max_new_tokens):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_ids = generated[0][input_ids.shape[1] :]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    del input_ids, attention_mask, generated
    torch.cuda.empty_cache()
    return answer_text


def normalize_answer_text(text):
    import re
    import string

    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def compute_text_f1(prediction, ground_truth):
    pred_tokens = normalize_answer_text(prediction).split()
    gt_tokens = normalize_answer_text(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counts = defaultdict(int)
    gt_counts = defaultdict(int)
    for tok in pred_tokens:
        pred_counts[tok] += 1
    for tok in gt_tokens:
        gt_counts[tok] += 1

    overlap = 0
    for tok, cnt in pred_counts.items():
        overlap += min(cnt, gt_counts.get(tok, 0))
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return float(2 * precision * recall / (precision + recall))


def format_ranked_nodes_for_tsv(ranked_nodes, topk=2):
    if not ranked_nodes:
        return ""
    return ",".join(str(int(x)) for x in ranked_nodes[: max(1, int(topk))])


def write_case_summary_tsv(results, out_path):
    strategy_names = sorted(
        {
            str(strategy_name)
            for result in results
            for strategy_name in result.get("strategy_results", {}).keys()
        }
    )

    header = [
        "doc_id",
        "query_idx",
        "meeting_id",
        "routing_unit_type",
        "relevant_units",
        "dominant_unit",
        "selected_unit_strategy",
        "selected_units",
        "selected_turn_hit",
        "selected_turn_recall",
        "selected_turn_precision",
        "selected_turn_f1",
        "full_answer_f1",
        "selected_answer_f1",
        "answer_f1_delta",
        "ctx_token_saving_pct",
    ]
    for strategy_name in strategy_names:
        header.extend(
            [
                f"{strategy_name}_top1",
                f"{strategy_name}_top2",
                f"{strategy_name}_top1_hit",
                f"{strategy_name}_top2_hit",
            ]
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for result in results:
            routing_eval = result.get("routing_eval", {})
            answer_eval = result.get("answer_eval") or {}
            qk_route_info = result.get("qk_route_info") or {}
            selected_units = qk_route_info.get("selected_topic_ids")
            if selected_units is None:
                selected_units = qk_route_info.get("selected_nodes", [])

            row = [
                str(int(result.get("doc_id", -1))),
                str(int(result.get("query_idx", -1))),
                str(result.get("meeting_id", "")),
                str(result.get("routing_unit_type", "")),
                ",".join(str(int(x)) for x in result.get("relevant_nodes", [])),
                str(int(result.get("dominant_relevant_node", -1))),
                str(routing_eval.get("selected_unit_strategy", "")),
                ",".join(str(int(x)) for x in (selected_units or [])),
                "1" if routing_eval.get("route_selected_turn_hit", False) else "0",
                f"{float(routing_eval.get('route_selected_turn_recall', 0.0)):.4f}",
                f"{float(routing_eval.get('route_selected_turn_precision', 0.0)):.4f}",
                f"{float(routing_eval.get('route_selected_turn_f1', 0.0)):.4f}",
                f"{float(answer_eval.get('full_answer_f1', 0.0)):.4f}",
                f"{float(answer_eval.get('selected_answer_f1', 0.0)):.4f}",
                f"{float(answer_eval.get('answer_f1_delta', 0.0)):.4f}",
                f"{100.0 * float(answer_eval.get('context_token_saving_ratio', 0.0)):.1f}",
            ]

            strategy_hits = routing_eval.get("strategy_hits", {})
            strategy_results = result.get("strategy_results", {})
            for strategy_name in strategy_names:
                sr = strategy_results.get(strategy_name, {})
                ranked_nodes = sr.get("ranked_nodes", [])
                hits = strategy_hits.get(strategy_name, {})
                row.extend(
                    [
                        str(int(sr.get("top_node", -1))) if sr else "",
                        format_ranked_nodes_for_tsv(ranked_nodes, topk=2),
                        "1" if hits.get("top1_any_relevant_hit", False) else "0",
                        "1" if hits.get("top2_any_relevant_hit", False) else "0",
                    ]
                )

            f.write("\t".join(row) + "\n")


def write_case_answer_log(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for result in results:
            answer_eval = result.get("answer_eval") or {}
            if not answer_eval:
                continue
            payload = {
                "doc_id": int(result.get("doc_id", -1)),
                "query_idx": int(result.get("query_idx", -1)),
                "meeting_id": result.get("meeting_id", ""),
                "query": result.get("query", ""),
                "gold_answer": answer_eval.get("gold_answer", ""),
                "full_answer": answer_eval.get("full_answer", ""),
                "selected_answer": answer_eval.get("selected_answer", ""),
                "full_answer_f1": float(answer_eval.get("full_answer_f1", 0.0)),
                "selected_answer_f1": float(answer_eval.get("selected_answer_f1", 0.0)),
                "answer_f1_delta": float(answer_eval.get("answer_f1_delta", 0.0)),
                "full_context_tokens": int(answer_eval.get("full_context_tokens", 0)),
                "selected_context_tokens": int(answer_eval.get("selected_context_tokens", 0)),
                "context_token_saving_ratio": float(answer_eval.get("context_token_saving_ratio", 0.0)),
                "selected_unit_strategy": result.get("routing_eval", {}).get("selected_unit_strategy", ""),
                "selected_topic_ids": result.get("qk_route_info", {}).get("selected_topic_ids", []),
                "selected_turns": result.get("routing_eval", {}).get("qk_selected_turns", []),
                "matched_turns": result.get("routing_eval", {}).get("qk_matched_turns", []),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_case_answer_markdown(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def md_block(text):
        return "~~~text\n" + (text or "").replace("~~~", "~~ ").rstrip() + "\n~~~"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# QMSum Answer Log\n\n")
        f.write("This file records one row per `(doc_id, query_idx)` sample.\n\n")
        for result in results:
            answer_eval = result.get("answer_eval") or {}
            if not answer_eval:
                continue
            routing_eval = result.get("routing_eval", {})
            f.write(f"## doc {int(result.get('doc_id', -1))} / query {int(result.get('query_idx', -1))}\n\n")
            f.write(f"- meeting_id: `{result.get('meeting_id', '')}`\n")
            f.write(f"- query: `{result.get('query', '')}`\n")
            f.write(f"- selected_unit_strategy: `{routing_eval.get('selected_unit_strategy', '')}`\n")
            f.write(
                f"- selected_topic_ids: `{','.join(str(x) for x in result.get('qk_route_info', {}).get('selected_topic_ids', []))}`\n"
            )
            f.write(
                f"- selected_turns: `{','.join(str(x) for x in routing_eval.get('qk_selected_turns', []))}`\n"
            )
            f.write(f"- matched_turns: `{','.join(str(x) for x in routing_eval.get('qk_matched_turns', []))}`\n")
            f.write(f"- full_answer_f1: `{float(answer_eval.get('full_answer_f1', 0.0)):.4f}`\n")
            f.write(f"- selected_answer_f1: `{float(answer_eval.get('selected_answer_f1', 0.0)):.4f}`\n")
            f.write(f"- answer_f1_delta: `{float(answer_eval.get('answer_f1_delta', 0.0)):.4f}`\n")
            f.write(f"- full_context_tokens: `{int(answer_eval.get('full_context_tokens', 0))}`\n")
            f.write(f"- selected_context_tokens: `{int(answer_eval.get('selected_context_tokens', 0))}`\n")
            f.write(
                f"- context_token_saving_ratio: `{100.0 * float(answer_eval.get('context_token_saving_ratio', 0.0)):.1f}%`\n\n"
            )
            f.write("### Gold Answer\n\n")
            f.write(md_block(answer_eval.get("gold_answer", "")) + "\n\n")
            f.write("### Full KV Answer\n\n")
            f.write(md_block(answer_eval.get("full_answer", "")) + "\n\n")
            f.write("### Selected KV Answer\n\n")
            f.write(md_block(answer_eval.get("selected_answer", "")) + "\n\n")


def build_output_suffix(args):
    parts = [f"N{args.num_nodes}", str(args.start_doc), str(args.end_doc)]
    if args.case_summary_tag:
        parts.append(str(args.case_summary_tag))
    return "_".join(parts)


def build_answer_eval(model, tokenizer, transcripts, query_text, gold_answer, selected_turns, max_new_tokens):
    full_prompt = build_qmsum_answer_prompt(
        transcripts,
        query_text,
        turn_indices=None,
    )
    selected_prompt = build_qmsum_answer_prompt(
        transcripts,
        query_text,
        turn_indices=selected_turns,
    )

    full_answer = generate_answer_text(
        model,
        tokenizer,
        full_prompt,
        max_new_tokens=max_new_tokens,
    )
    selected_answer = generate_answer_text(
        model,
        tokenizer,
        selected_prompt,
        max_new_tokens=max_new_tokens,
    )

    full_answer_f1 = compute_text_f1(full_answer, gold_answer)
    selected_answer_f1 = compute_text_f1(selected_answer, gold_answer)

    full_context_tokens = len(
        tokenizer(full_prompt, add_special_tokens=False).input_ids
    )
    selected_context_tokens = len(
        tokenizer(selected_prompt, add_special_tokens=False).input_ids
    )
    context_token_saving_ratio = 0.0
    if full_context_tokens > 0:
        context_token_saving_ratio = 1.0 - (
            selected_context_tokens / full_context_tokens
        )

    return {
        "gold_answer": gold_answer,
        "full_answer": full_answer,
        "selected_answer": selected_answer,
        "full_answer_f1": float(full_answer_f1),
        "selected_answer_f1": float(selected_answer_f1),
        "answer_f1_delta": float(selected_answer_f1 - full_answer_f1),
        "full_context_tokens": int(full_context_tokens),
        "selected_context_tokens": int(selected_context_tokens),
        "context_token_saving_ratio": float(context_token_saving_ratio),
        "selected_beats_or_matches_full": bool(selected_answer_f1 >= full_answer_f1),
    }


def main(args):
    from distributed_sim import (
        assign_turn_to_node,
        rank_scores,
        score_chunk_level,
        score_node_level,
        score_nodes_random,
        score_nodes_recency,
    )
    from experiment_chunk_split import _chunk_qk_scores_per_head
    from src.utils import define_model_and_tokenizer, split_kv

    print("=" * 70)
    print("QMSum routing simulation")
    print(f"  data: {args.data_path}")
    print(f"  routing_granularity: {args.routing_granularity}")
    if args.routing_granularity in ["node", "chunk"]:
        print(f"  num_nodes: {args.num_nodes}")
        print(f"  node_assignment_mode: {args.node_assignment_mode}")
    if args.routing_granularity == "chunk":
        print(f"  chunk_node_score_mode: {args.chunk_node_score_mode}")
    if args.routing_granularity == "hierarchical":
        print("  routing_units: dynamic topic nodes from qmsum topic_list")
        print(f"  hier_top_topics: {args.hier_top_topics}")
        print(f"  hier_top_strategy: {args.hier_top_strategy}")
        print(f"  hier_topic_score_mode: {args.hier_topic_score_mode}")
        print(
            f"  topic_embedding_source: {args.topic_embedding_source} "
            f"(prototype_turns={args.topic_prototype_turns})"
        )
        print(
            f"  topic_representation_template: {args.topic_representation_template}"
        )
        print(
            f"  topic_embedding_detail: {args.topic_embedding_turn_score_mode} "
            f"(topk={args.topic_embedding_topk}, label_weight={args.topic_label_weight:.2f})"
        )
        print(
            f"  lexical_detail: label_repeat={args.lexical_label_repeat}, "
            f"prf_top_topics={args.lexical_prf_top_topics}, "
            f"prf_terms={args.lexical_prf_terms}, "
            f"hybrid=({args.lexical_hybrid_embedding_weight:.2f}, "
            f"{args.lexical_hybrid_lexical_weight:.2f}), "
            f"rrf_k={args.rrf_k}"
        )
        if args.hier_top_strategy == "rerank":
            print(
                f"  rerank candidates: {args.rerank_candidate_topics}, "
                f"source: {args.rerank_source}, "
                f"weights: embedding={args.rerank_embedding_weight:.2f}, "
                f"qk={args.rerank_qk_weight:.2f}"
            )
    print("=" * 70)

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
    written_trace_paths = []

    def format_unit_id(unit_id, unit_label_for_display):
        prefix = "T" if unit_label_for_display == "topics" else "N"
        return f"{prefix}{int(unit_id)}"

    for doc_id in range(args.start_doc, args.end_doc):
        try:
            meeting = load_qmsum_sample(args.data_path, doc_id=doc_id)
        except IndexError:
            break

        transcripts = meeting.get("meeting_transcripts", [])
        if len(transcripts) < 4:
            continue

        specific_queries = meeting.get("specific_query_list", [])
        if args.max_queries_per_doc > 0:
            specific_queries = specific_queries[: args.max_queries_per_doc]

        topic_embedding_index = None
        if args.routing_granularity == "hierarchical":
            topic_embedding_index = build_topic_embedding_index_qmsum(
                meeting,
                transcripts,
                topic_embedding_source=args.topic_embedding_source,
                topic_prototype_turns=args.topic_prototype_turns,
                topic_representation_template=args.topic_representation_template,
            )

        prompt_text, turn_boundaries = build_qmsum_prompt(transcripts, tokenizer)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs.input_ids.cuda()
        total_tokens = input_ids.shape[1]

        if args.max_tokens > 0 and total_tokens > args.max_tokens:
            print(f"  [SKIP] doc_id={doc_id}: {total_tokens} tokens > {args.max_tokens}")
            del input_ids
            torch.cuda.empty_cache()
            continue

        try:
            with torch.no_grad():
                outputs = model.model(input_ids=input_ids, use_cache=True)
            kv_3d = tuple((layer[0][0], layer[1][0]) for layer in outputs.past_key_values)
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM SKIP] doc_id={doc_id}: {total_tokens} tokens")
            del input_ids
            torch.cuda.empty_cache()
            continue

        for query_idx, query_info in enumerate(specific_queries):
            query_text = query_info.get("query", "").strip()
            relevant_turns = spans_to_turn_set(query_info.get("relevant_text_span", []))
            if not query_text or not relevant_turns:
                continue

            query_ids = tokenizer(query_text, return_tensors="pt").input_ids.cuda()
            args._current_seed = doc_id * 1000 + query_idx

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
                eval_units_label = "nodes"
                relevant_node_counts = defaultdict(int)
                for turn_idx in relevant_turns:
                    node_id = assign_turn_to_node(
                        turn_idx,
                        len(transcripts),
                        num_nodes,
                        mode=args.node_assignment_mode,
                    )
                    relevant_node_counts[node_id] += 1
                relevant_units = sorted(relevant_node_counts.keys())
                dominant_unit = max(
                    sorted(relevant_node_counts.items()), key=lambda x: (x[1], -x[0])
                )[0]
                all_strategies = {"qk": node_scores}
                if args.baselines:
                    all_strategies["random"] = score_nodes_random(
                        num_nodes, seed=args._current_seed
                    )
                    all_strategies["recency"] = score_nodes_recency(node_turn_info, num_nodes)
                    all_strategies["embedding"] = score_nodes_embedding_qmsum(
                        query_text, node_turn_info, transcripts
                    )
                strategy_results = {
                    strat_name: rank_scores(scores)
                    for strat_name, scores in all_strategies.items()
                }
                if qk_route_info:
                    strategy_results["qk"]["route_info"] = qk_route_info
                selected_nodes = qk_route_info.get("selected_nodes", []) if qk_route_info else []
                selected_candidates = (
                    qk_route_info.get("selected_candidates", []) if qk_route_info else []
                )
                qk_values = [s for s in node_scores.values() if s > -1e8]
                total_route_units = num_nodes
                route_unit_info = {
                    "relevant_unit_counts": {
                        str(k): int(v) for k, v in sorted(relevant_node_counts.items())
                    }
                }
                topic_nodes = None
                turn_to_topic_ids = None
                qk_route_info_local = qk_route_info
            elif args.routing_granularity == "chunk":
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
                eval_units_label = "nodes"
                relevant_node_counts = defaultdict(int)
                for turn_idx in relevant_turns:
                    node_id = assign_turn_to_node(
                        turn_idx,
                        len(transcripts),
                        num_nodes,
                        mode=args.node_assignment_mode,
                    )
                    relevant_node_counts[node_id] += 1
                relevant_units = sorted(relevant_node_counts.keys())
                dominant_unit = max(
                    sorted(relevant_node_counts.items()), key=lambda x: (x[1], -x[0])
                )[0]
                all_strategies = {"qk": node_scores}
                if args.baselines:
                    all_strategies["random"] = score_nodes_random(
                        num_nodes, seed=args._current_seed
                    )
                    all_strategies["recency"] = score_nodes_recency(node_turn_info, num_nodes)
                    all_strategies["embedding"] = score_nodes_embedding_qmsum(
                        query_text, node_turn_info, transcripts
                    )
                strategy_results = {
                    strat_name: rank_scores(scores)
                    for strat_name, scores in all_strategies.items()
                }
                if qk_route_info:
                    strategy_results["qk"]["route_info"] = qk_route_info
                selected_nodes = qk_route_info.get("selected_nodes", []) if qk_route_info else []
                selected_candidates = (
                    qk_route_info.get("selected_candidates", []) if qk_route_info else []
                )
                qk_values = [s for s in node_scores.values() if s > -1e8]
                total_route_units = num_nodes
                route_unit_info = {
                    "relevant_unit_counts": {
                        str(k): int(v) for k, v in sorted(relevant_node_counts.items())
                    }
                }
                topic_nodes = None
                turn_to_topic_ids = None
                qk_route_info_local = qk_route_info
            elif args.routing_granularity == "hierarchical":
                (
                    topic_nodes,
                    topic_turn_info,
                    turn_to_topic_ids,
                    topic_scores,
                    strategy_results,
                    qk_route_info_local,
                    selected_candidates,
                ) = score_hierarchical_topic_chunk(
                    kv_3d,
                    turn_boundaries,
                    total_tokens,
                    query_text,
                    query_ids,
                    model,
                    scoring_layers,
                    split_kv,
                    _chunk_qk_scores_per_head,
                    meeting,
                    transcripts,
                    args,
                    topic_embedding_index=topic_embedding_index,
                )
                eval_units_label = "topics"
                relevant_topic_counts = defaultdict(int)
                for turn_idx in relevant_turns:
                    for topic_id in turn_to_topic_ids[turn_idx]:
                        relevant_topic_counts[topic_id] += 1
                relevant_units = sorted(relevant_topic_counts.keys())
                dominant_unit = max(
                    sorted(relevant_topic_counts.items()), key=lambda x: (x[1], -x[0])
                )[0]
                qk_values = [s for s in topic_scores.values() if s > -1e8]
                selected_nodes = (
                    qk_route_info_local.get("selected_topic_ids", [])
                    if qk_route_info_local
                    else []
                )
                total_route_units = len(topic_nodes)
                route_unit_info = {
                    "topic_nodes": [
                        {
                            "topic_id": int(topic["topic_id"]),
                            "label": topic["label"],
                            "num_turns": len(topic["turns"]),
                            "is_gap": bool(topic["is_gap"]),
                        }
                        for topic in topic_nodes
                    ],
                    "relevant_unit_counts": {
                        str(k): int(v) for k, v in sorted(relevant_topic_counts.items())
                    },
                }
            else:
                raise ValueError(f"Unsupported routing_granularity: {args.routing_granularity}")

            score_variance = float(torch.tensor(qk_values).var().item()) if len(qk_values) > 1 else 0.0
            score_range = float(max(qk_values) - min(qk_values)) if qk_values else 0.0
            selected_turn_metrics = compute_selected_turn_metrics(
                selected_candidates, relevant_turns
            )
            if args.routing_granularity == "hierarchical":
                transfer_unit_type = "topics"
                transfer_unit_field = "transfer_topic_id"
            else:
                transfer_unit_type = "nodes"
                transfer_unit_field = "node_id"
            transfer_accounting = build_transfer_accounting(
                selected_candidates,
                transfer_unit_type=transfer_unit_type,
                unit_field=transfer_unit_field,
            )
            if qk_route_info_local is not None:
                qk_route_info_local["transfer_accounting"] = transfer_accounting

            strategy_hits = {}
            for strat_name, sr in strategy_results.items():
                ranked_nodes = sr["ranked_nodes"]
                strategy_hits[strat_name] = {
                    "top1_any_relevant_hit": sr["top_node"] in relevant_units,
                    "top2_any_relevant_hit": any(node in relevant_units for node in ranked_nodes[:2]),
                    "top1_dominant_hit": sr["top_node"] == dominant_unit,
                    "top2_dominant_hit": dominant_unit in ranked_nodes[:2],
                }

            selected_node_hit = any(node in relevant_units for node in selected_nodes)
            answer_eval = None
            gold_answer = query_info.get("answer", "").strip()
            if args.eval_answers and gold_answer:
                answer_eval = build_answer_eval(
                    model,
                    tokenizer,
                    transcripts,
                    query_text,
                    gold_answer,
                    selected_turn_metrics["selected_turns"],
                    max_new_tokens=args.answer_max_new_tokens,
                )

            result_entry = {
                "doc_id": doc_id,
                "query_idx": query_idx,
                "meeting_id": meeting.get("meeting_id"),
                "query": query_text,
                "answer_preview": query_info.get("answer", "")[:200],
                "num_turns": len(transcripts),
                "total_tokens": int(total_tokens),
                "num_nodes": num_nodes,
                "num_route_units": int(total_route_units),
                "node_assignment_mode": args.node_assignment_mode,
                "relevant_turns": sorted(relevant_turns),
                "routing_unit_type": eval_units_label,
                "relevant_nodes": relevant_units,
                "dominant_relevant_node": int(dominant_unit),
                "relevant_node_counts": route_unit_info["relevant_unit_counts"],
                "qk_score_variance": score_variance,
                "qk_score_range": score_range,
                "strategy_results": strategy_results,
                "qk_route_info": qk_route_info_local,
                "route_unit_info": route_unit_info,
                "transfer_accounting": transfer_accounting,
                "answer_eval": answer_eval,
                "routing_eval": {
                    "selected_unit_strategy": (
                        args.hier_top_strategy
                        if args.routing_granularity == "hierarchical"
                        else "qk"
                    ),
                    "route_selected_unit_hit": selected_node_hit,
                    "route_selected_turn_hit": selected_turn_metrics["hit"],
                    "route_selected_turn_recall": selected_turn_metrics["recall"],
                    "route_selected_turn_precision": selected_turn_metrics["precision"],
                    "route_selected_turn_f1": selected_turn_metrics["f1"],
                    "route_selected_chunk_count": transfer_accounting["selected_chunk_count"],
                    "route_transfer_unit_count": transfer_accounting["unique_transfer_unit_count"],
                    "route_transfer_segment_count": transfer_accounting["transfer_segment_count"],
                    "qk_selected_node_hit": selected_node_hit,
                    "qk_selected_turn_hit": selected_turn_metrics["hit"],
                    "qk_selected_turn_recall": selected_turn_metrics["recall"],
                    "qk_selected_turn_precision": selected_turn_metrics["precision"],
                    "qk_selected_turn_f1": selected_turn_metrics["f1"],
                    "qk_selected_turns": selected_turn_metrics["selected_turns"],
                    "qk_matched_turns": selected_turn_metrics["matched_turns"],
                    "strategy_hits": strategy_hits,
                },
            }
            results.append(result_entry)

            if should_trace_case(args, doc_id, query_idx):
                functions_used = [
                    {
                        "function": "load_qmsum_sample",
                        "purpose": "Load one meeting record from the prepared QMSum jsonl file.",
                        "key_outputs": {
                            "meeting_id": meeting.get("meeting_id"),
                            "num_transcript_turns": len(transcripts),
                            "num_topics_raw": len(meeting.get("topic_list", [])),
                            "num_specific_queries": len(meeting.get("specific_query_list", [])),
                        },
                    },
                    {
                        "function": "build_qmsum_prompt",
                        "purpose": "Flatten meeting turns into one prompt string and record token boundaries per turn.",
                        "key_outputs": {
                            "prompt_char_len": len(prompt_text),
                            "num_turn_boundaries": len(turn_boundaries),
                            "total_tokens": int(total_tokens),
                        },
                    },
                    {
                        "function": "define_model_and_tokenizer / model.model(use_cache=True)",
                        "purpose": "Run the whole meeting once through the base model and cache KV tensors.",
                        "key_outputs": {
                            "num_model_layers": len(kv_3d),
                            "kv_token_len": int(total_tokens),
                        },
                    },
                ]

                trace_payload = {
                    "config": {
                        "routing_granularity": args.routing_granularity,
                        "hier_top_topics": args.hier_top_topics,
                        "hier_top_strategy": args.hier_top_strategy,
                        "hier_topic_score_mode": args.hier_topic_score_mode,
                        "topic_embedding_turn_score_mode": args.topic_embedding_turn_score_mode,
                        "topic_embedding_topk": args.topic_embedding_topk,
                        "topic_label_weight": args.topic_label_weight,
                        "route_chunk_size": args.route_chunk_size,
                        "route_top_k": args.route_top_k,
                        "route_per_head": bool(args.route_per_head),
                        "route_neighbor_expand": int(args.route_neighbor_expand),
                    },
                    "target": {"doc_id": int(doc_id), "query_idx": int(query_idx)},
                    "meeting": {
                        "meeting_id": meeting.get("meeting_id"),
                        "num_turns": len(transcripts),
                        "num_topics_raw": len(meeting.get("topic_list", [])),
                        "num_specific_queries": len(meeting.get("specific_query_list", [])),
                        "first_turns_preview": [
                            {
                                "turn_idx": idx,
                                "speaker": turn.get("speaker", "").strip(),
                                "content_preview": preview_text(turn.get("content", ""), limit=120),
                            }
                            for idx, turn in enumerate(transcripts[:5])
                        ],
                    },
                    "query": {
                        "text": query_text,
                        "answer_preview": preview_text(query_info.get("answer", ""), limit=200),
                        "relevant_turns": sorted(relevant_turns),
                        "relevant_turn_count": len(relevant_turns),
                        "relevant_topic_ids": [int(x) for x in relevant_units],
                    },
                    "functions_used": functions_used,
                    "result_entry_preview": {
                        "routing_unit_type": result_entry["routing_unit_type"],
                        "qk_score_variance": result_entry["qk_score_variance"],
                        "qk_score_range": result_entry["qk_score_range"],
                    },
                }

                if args.routing_granularity == "hierarchical":
                    functions_used.extend(
                        [
                            {
                            "function": "build_topic_nodes",
                            "purpose": "Convert QMSum topic spans into topic nodes and map turns to topic ids.",
                            "key_outputs": {
                                "num_topic_nodes": len(topic_nodes),
                                "topic_ids": [int(t["topic_id"]) for t in topic_nodes],
                            },
                        },
                        {
                            "function": "build_topic_embedding_index_qmsum",
                            "purpose": "Build coarse topic representations once per meeting for topic-level routing.",
                            "key_outputs": {
                                "topic_embedding_source": args.topic_embedding_source,
                                "topic_prototype_turns": args.topic_prototype_turns,
                                "topic_representation_template": args.topic_representation_template,
                                "num_topic_repr_entries": sum(
                                    len(v)
                                    for v in topic_embedding_index.get("topic_repr_entries", {}).values()
                                )
                                if topic_embedding_index is not None
                                else 0,
                            },
                        },
                        {
                            "function": "score_topics_embedding_qmsum",
                            "purpose": "Compute top-level topic scores from either precomputed topic representations or online turn-aware aggregation.",
                            "key_outputs": {
                                    "embedding_top_topic": int(strategy_results["embedding"]["top_node"])
                                    if "embedding" in strategy_results
                                    else None,
                                    "embedding_ranked_topics_top5": [
                                        int(x)
                                        for x in strategy_results.get("embedding", {}).get(
                                            "ranked_nodes", []
                                        )[:5]
                                    ],
                                },
                            },
                            {
                                "function": "build_hierarchical_candidates",
                                "purpose": "Split turns into chunk candidates and attach each chunk to topic ids.",
                                "key_outputs": {
                                    "num_candidates": int(qk_route_info_local.get("num_candidates", 0))
                                    if qk_route_info_local
                                    else 0,
                                    "num_candidates_after_topic_filter": int(
                                        qk_route_info_local.get("num_candidates_after_topic_filter", 0)
                                    )
                                    if qk_route_info_local
                                    else 0,
                                },
                            },
                            {
                                "function": "score_hierarchical_topic_chunk",
                                "purpose": "Use Q-K scores to rank chunk candidates inside the selected topic.",
                                "key_outputs": {
                                    "selected_topic_ids": qk_route_info_local.get("selected_topic_ids", [])
                                    if qk_route_info_local
                                    else [],
                                    "num_selected_candidates": int(
                                        qk_route_info_local.get("num_selected_candidates", 0)
                                    )
                                    if qk_route_info_local
                                    else 0,
                                    "selected_token_count": int(
                                        qk_route_info_local.get("selected_token_count", 0)
                                    )
                                    if qk_route_info_local
                                    else 0,
                                },
                            },
                            {
                                "function": "compute_selected_turn_metrics",
                                "purpose": "Compare selected chunk turns against ground-truth relevant turns.",
                                "key_outputs": {
                                    "matched_turns": selected_turn_metrics["matched_turns"],
                                    "recall": selected_turn_metrics["recall"],
                                    "precision": selected_turn_metrics["precision"],
                                    "f1": selected_turn_metrics["f1"],
                                },
                            },
                            {
                                "function": "build_transfer_accounting",
                                "purpose": "Summarize how many chunks / segments would be transferred in a distributed setup.",
                                "key_outputs": {
                                    "selected_chunk_count": transfer_accounting["selected_chunk_count"],
                                    "transfer_segment_count": transfer_accounting["transfer_segment_count"],
                                    "coalescing_gain_transfer": transfer_accounting[
                                        "coalescing_gain_transfer"
                                    ],
                                },
                            },
                        ]
                    )

                    trace_payload["topic_nodes"] = {
                        "all_topic_nodes": route_unit_info.get("topic_nodes", []),
                        "relevant_topic_counts": route_unit_info.get("relevant_unit_counts", {}),
                        "topic_repr_turns": (
                            topic_embedding_index.get("topic_repr_turns", {})
                            if topic_embedding_index is not None
                            else {}
                        ),
                        "topic_repr_entries": (
                            {
                                str(topic_id): [
                                    {
                                        "entry_type": entry.get("entry_type"),
                                        "text_preview": preview_text(entry.get("text", ""), limit=120),
                                        "turn_idx": entry.get("turn_idx"),
                                    }
                                    for entry in entry_list
                                ]
                                for topic_id, entry_list in topic_embedding_index.get(
                                    "topic_repr_entries", {}
                                ).items()
                            }
                            if topic_embedding_index is not None
                            else {}
                        ),
                    }
                    trace_payload["top_level_routing"] = {
                        "strategy_rankings_top5": {
                            strat_name: [int(x) for x in sr.get("ranked_nodes", [])[:5]]
                            for strat_name, sr in strategy_results.items()
                        },
                        "strategy_top_scores": {
                            strat_name: {
                                "top_node": int(sr["top_node"]),
                                "top_score": float(sr["sorted_scores"][0][1])
                                if sr.get("sorted_scores")
                                else None,
                            }
                            for strat_name, sr in strategy_results.items()
                        },
                        "selected_topic_ids": qk_route_info_local.get("selected_topic_ids", [])
                        if qk_route_info_local
                        else [],
                        "selected_topics": qk_route_info_local.get("selected_topics", [])
                        if qk_route_info_local
                        else [],
                    }
                    trace_payload["chunk_routing"] = {
                        "num_candidates": qk_route_info_local.get("num_candidates", 0)
                        if qk_route_info_local
                        else 0,
                        "num_candidates_after_topic_filter": qk_route_info_local.get(
                            "num_candidates_after_topic_filter", 0
                        )
                        if qk_route_info_local
                        else 0,
                        "num_selected_candidates": qk_route_info_local.get(
                            "num_selected_candidates", 0
                        )
                        if qk_route_info_local
                        else 0,
                        "selected_candidates_preview": qk_route_info_local.get(
                            "selected_candidates", []
                        )
                        if qk_route_info_local
                        else [],
                    }
                    trace_payload["evaluation"] = {
                        "selected_topic_hit": bool(selected_node_hit),
                        "selected_turn_hit": bool(selected_turn_metrics["hit"]),
                        "matched_turns": selected_turn_metrics["matched_turns"],
                        "selected_turns": selected_turn_metrics["selected_turns"],
                        "turn_recall": float(selected_turn_metrics["recall"]),
                        "turn_precision": float(selected_turn_metrics["precision"]),
                        "turn_f1": float(selected_turn_metrics["f1"]),
                    }
                    if answer_eval:
                        trace_payload["answer_eval"] = {
                            "gold_answer_preview": preview_text(answer_eval.get("gold_answer", ""), limit=240),
                            "full_answer_preview": preview_text(answer_eval.get("full_answer", ""), limit=240),
                            "selected_answer_preview": preview_text(answer_eval.get("selected_answer", ""), limit=240),
                            "full_answer_f1": float(answer_eval.get("full_answer_f1", 0.0)),
                            "selected_answer_f1": float(answer_eval.get("selected_answer_f1", 0.0)),
                            "answer_f1_delta": float(answer_eval.get("answer_f1_delta", 0.0)),
                            "full_context_tokens": int(answer_eval.get("full_context_tokens", 0)),
                            "selected_context_tokens": int(answer_eval.get("selected_context_tokens", 0)),
                            "context_token_saving_ratio": float(
                                answer_eval.get("context_token_saving_ratio", 0.0)
                            ),
                        }
                    trace_payload["transfer_accounting"] = transfer_accounting

                    json_path, md_path = write_case_trace(
                        trace_payload,
                        output_dir=args.trace_output_dir,
                        output_format=args.trace_output_format,
                    )
                    written_trace_paths.append((json_path, md_path))

            if len(results) <= 3:
                print("\n" + "-" * 60)
                print(f"[doc {doc_id} / query {query_idx}] {len(transcripts)} turns, {total_tokens} tokens")
                print(f"  Query: {query_text[:120]!r}")
                print(f"  Relevant {eval_units_label}: {relevant_units}")
                print(
                    f"  Dominant {eval_units_label[:-1] if eval_units_label.endswith('s') else eval_units_label}: "
                    f"{format_unit_id(dominant_unit, eval_units_label)}"
                )
                print(f"  Relevant turns: {sorted(relevant_turns)[:12]}")
                print(
                    f"  Q-K top {eval_units_label[:-1] if eval_units_label.endswith('s') else eval_units_label}: "
                    f"{format_unit_id(strategy_results['qk']['top_node'], eval_units_label)}"
                )
                if qk_route_info_local:
                    if args.routing_granularity == "hierarchical":
                        print(
                            f"  selected topics: {qk_route_info_local['selected_topic_ids']}, "
                            f"chunks={qk_route_info_local['num_selected_candidates']}/"
                            f"{qk_route_info_local['num_candidates_after_topic_filter']}"
                        )
                        if args.hier_top_strategy == "rerank":
                            print(
                                f"  rerank candidates: "
                                f"{qk_route_info_local.get('rerank_candidate_topic_ids', [])}"
                            )
                    else:
                        print(
                            f"  selected chunks: {qk_route_info_local['num_selected_candidates']}/"
                            f"{qk_route_info_local['num_candidates']}, nodes={qk_route_info_local['selected_nodes']}"
                        )
                    print(
                        f"  matched turns: {selected_turn_metrics['matched_turns'][:8]}, "
                        f"recall={selected_turn_metrics['recall']:.3f}"
                    )
                    print(
                        f"  transfer units={transfer_accounting['unique_transfer_unit_count']}, "
                        f"segments={transfer_accounting['transfer_segment_count']}, "
                        f"selected_tokens={transfer_accounting['selected_token_count']}"
                    )
                if answer_eval:
                    print(
                        f"  answer F1 full={answer_eval['full_answer_f1']:.3f}, "
                        f"selected={answer_eval['selected_answer_f1']:.3f}, "
                        f"ctx saving={100 * answer_eval['context_token_saving_ratio']:.1f}%"
                    )

            del query_ids
            torch.cuda.empty_cache()

        del input_ids, kv_3d
        torch.cuda.empty_cache()

    if len(results) == 0:
        print("No results")
        return

    summary_payload = build_summary_payload(results, args, num_nodes)
    print_summary(summary_payload, args, num_nodes)

    if not args.no_main_output:
        output_suffix = build_output_suffix(args)
        out_path = f"outputs/qmsum_sim_{output_suffix}.json"
        case_tsv_path = f"outputs/qmsum_case_summary_{output_suffix}.tsv"
        answer_log_path = f"outputs/qmsum_answer_log_{output_suffix}.jsonl"
        answer_md_path = f"outputs/qmsum_answer_log_{output_suffix}.md"
        os.makedirs("outputs", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "dataset": "qmsum",
                        "data_path": args.data_path,
                        "num_nodes": num_nodes,
                        "node_assignment_mode": args.node_assignment_mode,
                        "scoring_layers": args.scoring_layers,
                        "routing_granularity": args.routing_granularity,
                        "hier_top_topics": args.hier_top_topics,
                        "hier_top_strategy": args.hier_top_strategy,
                        "hier_topic_score_mode": args.hier_topic_score_mode,
                        "hier_topic_topk": args.hier_topic_topk,
                        "topic_embedding_turn_score_mode": args.topic_embedding_turn_score_mode,
                        "topic_embedding_topk": args.topic_embedding_topk,
                        "topic_label_weight": args.topic_label_weight,
                        "rerank_candidate_topics": args.rerank_candidate_topics,
                        "rerank_source": args.rerank_source,
                        "rerank_embedding_weight": args.rerank_embedding_weight,
                        "rerank_qk_weight": args.rerank_qk_weight,
                        "route_chunk_size": args.route_chunk_size,
                        "route_top_k": args.route_top_k,
                        "route_per_head": args.route_per_head,
                        "route_neighbor_expand": args.route_neighbor_expand,
                        "topic_embedding_source": args.topic_embedding_source,
                        "topic_prototype_turns": args.topic_prototype_turns,
                        "topic_representation_template": args.topic_representation_template,
                        "lexical_label_repeat": args.lexical_label_repeat,
                        "lexical_prf_top_topics": args.lexical_prf_top_topics,
                        "lexical_prf_terms": args.lexical_prf_terms,
                        "lexical_hybrid_embedding_weight": args.lexical_hybrid_embedding_weight,
                        "lexical_hybrid_lexical_weight": args.lexical_hybrid_lexical_weight,
                        "rrf_k": args.rrf_k,
                        "chunk_node_score_mode": args.chunk_node_score_mode,
                        "qk_aggregation": args.qk_aggregation,
                        "qk_topk": args.qk_topk,
                        "baselines": args.baselines,
                        "max_queries_per_doc": args.max_queries_per_doc,
                    },
                    "summary": {
                        "n": summary_payload["n"],
                        "avg_turns": summary_payload["avg_turns"],
                        "routing_unit_type": summary_payload["unit_label"],
                        "avg_relevant_nodes": summary_payload["avg_relevant_nodes"],
                        "all_node_relevant_span_rate": summary_payload["all_node_covered"] / summary_payload["n"]
                        if summary_payload["n"] > 0
                        else 0,
                        "qk_avg_score_variance": summary_payload["avg_variance"],
                        "qk_avg_score_range": summary_payload["avg_range"],
                        "qk_high_variance_ratio": summary_payload["high_var"] / summary_payload["n"]
                        if summary_payload["n"] > 0
                        else 0,
                        "per_strategy": summary_payload["per_strategy"],
                        "qk_selected_node_hit_rate": summary_payload["selected_node_hits"] / summary_payload["n"],
                        "qk_selected_turn_hit_rate": summary_payload["selected_turn_hits"] / summary_payload["n"],
                        "qk_avg_selected_turn_recall": summary_payload["avg_selected_turn_recall"],
                        "qk_avg_selected_turn_precision": summary_payload["avg_selected_turn_precision"],
                        "qk_avg_selected_turn_f1": summary_payload["avg_selected_turn_f1"],
                        "avg_selected_chunk_count": summary_payload["avg_selected_chunks"],
                        "avg_selected_token_count": summary_payload["avg_selected_tokens"],
                        "avg_transfer_unit_count": summary_payload["avg_transfer_units"],
                        "avg_transfer_segment_count": summary_payload["avg_transfer_segments"],
                        "avg_global_contiguous_segment_count": summary_payload["avg_global_segments"],
                        "avg_chunks_per_transfer_segment": summary_payload["avg_chunks_per_transfer_segment"],
                        "avg_transfer_coalescing_gain": summary_payload["avg_coalescing_gain"],
                        "answer_summary": summary_payload.get("answer_summary"),
                    },
                    "details": results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved to {out_path}")
        write_case_summary_tsv(results, case_tsv_path)
        print(f"Case summary TSV saved to {case_tsv_path}")
        if args.eval_answers:
            write_case_answer_log(results, answer_log_path)
            print(f"Answer log saved to {answer_log_path}")
            write_case_answer_markdown(results, answer_md_path)
            print(f"Answer markdown saved to {answer_md_path}")

    for json_path, md_path in written_trace_paths:
        if json_path:
            print(f"Trace JSON saved to {json_path}")
        if md_path:
            print(f"Trace Markdown saved to {md_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QMSum routing simulation")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_gpu_memory", type=int, default=48)
    p.add_argument("--num_nodes", type=int, default=4)
    p.add_argument("--scoring_layers", type=str, default=None)
    p.add_argument("--start_doc", type=int, default=0)
    p.add_argument("--end_doc", type=int, default=10)
    p.add_argument("--max_queries_per_doc", type=int, default=2)
    p.add_argument("--baselines", action="store_true", default=False)
    p.add_argument(
        "--node_assignment_mode",
        type=str,
        default="contiguous",
        choices=["round_robin", "contiguous"],
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="Skip documents longer than this token count. Use 0 to disable the limit.",
    )
    p.add_argument(
        "--routing_granularity",
        type=str,
        default="chunk",
        choices=["node", "chunk", "hierarchical"],
    )
    p.add_argument("--hier_top_topics", type=int, default=1)
    p.add_argument(
        "--hier_top_strategy",
        type=str,
        default="embedding",
        choices=[
            "qk",
            "embedding",
            "rerank",
            "random",
            "recency",
            "lexical",
            "lexical_prf",
            "lexical_hybrid",
            "rrf",
        ],
    )
    p.add_argument(
        "--hier_topic_score_mode",
        type=str,
        default="sum",
        choices=["sum", "max", "topk_mean"],
    )
    p.add_argument("--hier_topic_topk", type=int, default=4)
    p.add_argument(
        "--topic_embedding_turn_score_mode",
        type=str,
        default="topk_mean",
        choices=["mean", "topk_mean"],
    )
    p.add_argument("--topic_embedding_topk", type=int, default=3)
    p.add_argument("--topic_label_weight", type=float, default=0.35)
    p.add_argument(
        "--topic_embedding_source",
        type=str,
        default="precomputed_topic_text",
        choices=["online_turns", "precomputed_topic_text"],
    )
    p.add_argument("--topic_prototype_turns", type=int, default=3)
    p.add_argument(
        "--topic_representation_template",
        type=str,
        default="basic",
        choices=["basic", "enhanced"],
    )
    p.add_argument("--lexical_label_repeat", type=int, default=3)
    p.add_argument("--lexical_prf_top_topics", type=int, default=1)
    p.add_argument("--lexical_prf_terms", type=int, default=4)
    p.add_argument("--lexical_hybrid_embedding_weight", type=float, default=1.0)
    p.add_argument("--lexical_hybrid_lexical_weight", type=float, default=0.7)
    p.add_argument("--rrf_k", type=int, default=60)
    p.add_argument("--rerank_candidate_topics", type=int, default=3)
    p.add_argument(
        "--rerank_source",
        type=str,
        default="qk_topic",
        choices=["qk_topic", "candidate_turns"],
    )
    p.add_argument("--rerank_embedding_weight", type=float, default=1.0)
    p.add_argument("--rerank_qk_weight", type=float, default=0.5)
    p.add_argument("--route_chunk_size", type=int, default=128)
    p.add_argument("--route_top_k", type=int, default=4)
    p.add_argument("--route_per_head", action="store_true", default=False)
    p.add_argument("--route_neighbor_expand", type=int, default=0)
    p.add_argument(
        "--chunk_node_score_mode",
        type=str,
        default="selected_count",
        choices=["selected_sum", "selected_count", "selected_max", "all_chunk_max"],
    )
    p.add_argument("--qk_aggregation", type=str, default="mean", choices=["mean", "max", "topk_mean"])
    p.add_argument("--qk_topk", type=int, default=4)
    p.add_argument("--trace_doc_id", type=int, default=-1)
    p.add_argument("--trace_query_idx", type=int, default=-1)
    p.add_argument("--trace_output_dir", type=str, default="outputs/qmsum_trace")
    p.add_argument(
        "--trace_output_format",
        type=str,
        default="md",
        choices=["md", "json", "both"],
    )
    p.add_argument("--eval_answers", action="store_true", default=False)
    p.add_argument("--answer_max_new_tokens", type=int, default=96)
    p.add_argument("--no_main_output", action="store_true", default=False)
    p.add_argument("--case_summary_tag", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if args.data_path is None:
        args.data_path = os.path.join(
            os.path.dirname(__file__), "data", "qmsum_structured", "train.jsonl"
        )

    if not os.path.exists(args.data_path):
        print(f"Data file does not exist: {args.data_path}")
        sys.exit(1)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    main(args)
