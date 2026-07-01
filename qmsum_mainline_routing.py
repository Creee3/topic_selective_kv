import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from qmsum_data import build_topic_nodes


VALID_NODE_ASSIGNMENT_MODES = {"contiguous", "round_robin", "manual"}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}


def tokenize_lexical_text(text):
    tokens = []
    current = []
    for ch in (text or "").lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                token = "".join(current)
                if token and token not in STOPWORDS:
                    tokens.append(token)
                current = []
    if current:
        token = "".join(current)
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def jaccard_overlap(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def diversify_selected_candidates(
    candidates,
    transcripts,
    target_keep_count,
    min_keep_count,
    max_similarity=0.8,
):
    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda x: -float(x["score"]))
    selected = []
    selected_token_sets = []
    min_keep_count = max(1, int(min_keep_count))
    target_keep_count = max(min_keep_count, int(target_keep_count))

    for idx, cand in enumerate(ranked):
        turn_idx = int(cand["turn_idx"])
        if 0 <= turn_idx < len(transcripts):
            turn = transcripts[turn_idx]
            speaker = turn.get("speaker", "").strip() or "Speaker"
            content = turn.get("content", "").strip()
            text = f"{speaker}: {content}"
        else:
            text = ""

        cand_tokens = tokenize_lexical_text(text)
        too_similar = False
        for prev_tokens in selected_token_sets:
            if jaccard_overlap(cand_tokens, prev_tokens) >= float(max_similarity):
                too_similar = True
                break
        if too_similar and len(selected) >= min_keep_count:
            continue

        selected.append(cand)
        selected_token_sets.append(cand_tokens)
        if len(selected) >= target_keep_count:
            break

    if not selected:
        return ranked[:target_keep_count]
    return selected


def classify_query_budget_type(query_text):
    """Heuristic query type used only for dynamic fine-stage budget control."""
    text = (query_text or "").lower()
    detail_terms = [
        "decision",
        "decide",
        "what's the decision",
        "what is the decision",
        "what did",
        "what could",
        "what should",
        "who",
        "when",
        "where",
        "how many",
        "number",
        "think",
        "opinion",
        "suggest",
        "agree",
        "in regard to",
    ]
    summary_terms = [
        "summarize",
        "summary",
        "overview",
        "presentation",
        "discussion",
        "discussed",
        "main points",
        "what happened",
    ]

    # Specific question intents should win over broad words like "discussion".
    if any(term in text for term in detail_terms):
        return "detail"
    if any(term in text for term in summary_terms):
        return "summary"
    return "balanced"


def resolve_route_top_k_for_query(args, query_text):
    """Return the effective chunk budget for this query."""
    base_top_k = max(1, int(args.route_top_k))
    query_type = classify_query_budget_type(query_text)

    if not getattr(args, "dynamic_route_budget", False):
        return base_top_k, query_type, "fixed"

    if query_type == "summary":
        budget = int(getattr(args, "dynamic_summary_top_k", base_top_k))
    elif query_type == "detail":
        budget = int(getattr(args, "dynamic_detail_top_k", base_top_k))
    else:
        budget = int(getattr(args, "dynamic_balanced_top_k", base_top_k))

    return max(1, budget), query_type, "dynamic"


def maybe_apply_adaptive_topic_rescue(
    lexical_ranked_topics,
    lexical_result,
    topic_nodes,
    args,
):
    """Optionally add a second topic when lexical top-1 is uncertain."""
    base_k = min(max(1, int(args.hier_top_topics)), len(topic_nodes))
    selected_topic_ids = list(lexical_ranked_topics[:base_k])
    info = {
        "enabled": bool(getattr(args, "adaptive_topic_rescue", False)),
        "triggered": False,
        "base_k": int(base_k),
        "final_k": int(len(selected_topic_ids)),
        "max_topics": int(getattr(args, "adaptive_topic_rescue_max_topics", 2)),
        "reason": "disabled",
        "top1_topic": int(lexical_ranked_topics[0]) if lexical_ranked_topics else -1,
        "top2_topic": (
            int(lexical_ranked_topics[base_k])
            if len(lexical_ranked_topics) > base_k
            else -1
        ),
        "top1_score": 0.0,
        "top2_score": 0.0,
        "score_gap": 0.0,
        "score_span": 0.0,
        "gap_ratio": 0.0,
        "top1_ratio": 0.0,
        "margin_ratio": float(
            getattr(args, "adaptive_topic_rescue_margin_ratio", 0.12)
        ),
        "min_top1_ratio": float(
            getattr(args, "adaptive_topic_rescue_min_top1_ratio", 1.15)
        ),
        "min_score": float(getattr(args, "adaptive_topic_rescue_min_score", 0.0)),
    }
    if not info["enabled"]:
        return selected_topic_ids, info
    if base_k >= len(lexical_ranked_topics):
        info["reason"] = "base_k_covers_all_topics"
        return selected_topic_ids, info

    max_topics = min(
        max(base_k, int(getattr(args, "adaptive_topic_rescue_max_topics", 2))),
        len(lexical_ranked_topics),
    )
    info["max_topics"] = int(max_topics)
    if max_topics <= base_k:
        info["reason"] = "max_topics_not_above_base_k"
        return selected_topic_ids, info

    scores = lexical_result.get("scores", {}) if lexical_result else {}
    ranked_scores = [
        float(scores.get(int(topic_id), 0.0) or 0.0)
        for topic_id in lexical_ranked_topics
    ]
    if len(ranked_scores) < base_k + 1:
        info["reason"] = "no_extra_topic"
        return selected_topic_ids, info

    top1_score = float(ranked_scores[0])
    top2_score = float(ranked_scores[base_k])
    max_score = max(ranked_scores) if ranked_scores else 0.0
    min_ranked_score = min(ranked_scores) if ranked_scores else 0.0
    score_span = max(1e-9, float(max_score - min_ranked_score))
    score_gap = float(top1_score - top2_score)
    gap_ratio = score_gap / score_span
    positive_top1 = max(0.0, top1_score)
    positive_top2 = max(0.0, top2_score)
    top1_ratio = (
        (positive_top1 + 1e-9) / (positive_top2 + 1e-9)
        if positive_top2 > 0.0
        else float("inf")
    )
    margin_ratio = max(0.0, float(info["margin_ratio"]))
    min_top1_ratio = max(1.0, float(info["min_top1_ratio"]))
    min_score = float(info["min_score"])

    info.update(
        {
            "reason": "confident_top1",
            "top1_score": top1_score,
            "top2_score": top2_score,
            "score_gap": score_gap,
            "score_span": score_span,
            "gap_ratio": gap_ratio,
            "top1_ratio": top1_ratio if np.isfinite(top1_ratio) else 1e9,
            "margin_ratio": margin_ratio,
            "min_top1_ratio": min_top1_ratio,
            "min_score": min_score,
        }
    )

    should_rescue = False
    reasons = []
    if top2_score < min_score:
        info["reason"] = "top2_below_min_score"
    else:
        if gap_ratio <= margin_ratio:
            should_rescue = True
            reasons.append("small_gap")
        if top1_ratio <= min_top1_ratio:
            should_rescue = True
            reasons.append("low_top1_ratio")

    if should_rescue:
        selected_topic_ids = list(lexical_ranked_topics[:max_topics])
        info["triggered"] = True
        info["final_k"] = int(len(selected_topic_ids))
        info["reason"] = "+".join(reasons) if reasons else "uncertain_top1"

    return selected_topic_ids, info


def order_candidates_for_answer(candidates, mode):
    """Order selected evidence chunks before deriving answer turns."""
    if mode == "qk":
        return sorted(
            candidates,
            key=lambda c: (-float(c.get("score", 0.0)), int(c["start_t"]), int(c["turn_idx"])),
        )
    if mode == "qk_then_time":
        top_count = min(4, len(candidates))
        qk_prefix = sorted(candidates, key=lambda c: -float(c.get("score", 0.0)))[:top_count]
        prefix_ids = {id(c) for c in qk_prefix}
        tail = [c for c in candidates if id(c) not in prefix_ids]
        return qk_prefix + sorted(tail, key=lambda c: (int(c["start_t"]), int(c["turn_idx"])))
    return sorted(candidates, key=lambda c: (int(c["start_t"]), int(c["turn_idx"])))


def candidate_turns_in_order(candidates):
    ordered_turns = []
    seen = set()
    for cand in candidates:
        turn_idx = int(cand["turn_idx"])
        if turn_idx in seen:
            continue
        ordered_turns.append(turn_idx)
        seen.add(turn_idx)
    return ordered_turns


def get_turn_text(transcripts, turn_idx):
    if 0 <= int(turn_idx) < len(transcripts):
        turn = transcripts[int(turn_idx)]
        speaker = turn.get("speaker", "").strip() or "Speaker"
        content = turn.get("content", "").strip()
        return speaker, f"{speaker}: {content}"
    return "Speaker", ""


def normalize_numeric_map(raw_values):
    finite_values = [
        float(value)
        for value in raw_values.values()
        if np.isfinite(float(value)) and float(value) > -1e8
    ]
    if not finite_values:
        return {int(key): 0.0 for key in raw_values}

    lo = min(finite_values)
    hi = max(finite_values)
    span = max(hi - lo, 1e-6)
    normalized = {}
    for key, value in raw_values.items():
        value = float(value)
        if not np.isfinite(value) or value <= -1e8:
            normalized[int(key)] = 0.0
        else:
            normalized[int(key)] = float((value - lo) / span)
    return normalized


def compute_candidate_head_vote_scores(per_head_scores, top_k):
    if not per_head_scores:
        return {}

    score_matrix = np.stack(per_head_scores, axis=0)
    num_candidates, num_heads = score_matrix.shape
    if num_candidates <= 0 or num_heads <= 0:
        return {}

    k = min(max(1, int(top_k)), num_candidates)
    vote_counts = np.zeros(num_candidates, dtype=np.float32)
    for head_idx in range(num_heads):
        top_local = np.argsort(score_matrix[:, head_idx])[-k:]
        vote_counts[top_local] += 1.0

    denom = max(1.0, float(num_heads))
    return {
        int(local_idx): float(vote_counts[local_idx] / denom)
        for local_idx in range(num_candidates)
    }


def select_candidates_chunk_topk(
    scored_candidates,
    per_head_scores,
    effective_route_top_k,
    use_per_head,
):
    if not scored_candidates:
        return [], {}

    k = min(int(effective_route_top_k), len(scored_candidates))
    debug = {"selection_mode": "chunk_topk", "target_k": int(k)}
    if use_per_head and per_head_scores:
        score_matrix = np.stack(per_head_scores, axis=0)
        selected_local_ids = set()
        for head_idx in range(score_matrix.shape[1]):
            top_local = np.argsort(score_matrix[:, head_idx])[-k:]
            selected_local_ids.update(int(i) for i in top_local)
        selected_candidates = [scored_candidates[i] for i in selected_local_ids]
        selected_candidates = sorted(
            selected_candidates,
            key=lambda x: -float(x["score"]),
        )[:k]
        debug["per_head_candidate_pool"] = int(len(selected_local_ids))
    else:
        selected_candidates = sorted(
            scored_candidates,
            key=lambda x: -float(x["score"]),
        )[:k]
    return selected_candidates, debug


def select_candidates_turn_utility(
    scored_candidates,
    per_head_scores,
    candidate_prefilter_scores,
    effective_route_top_k,
    args,
):
    """Select evidence at turn granularity using chunk-level Q-K as utility.

    The QMSum supervision is turn-level.  Raw chunk top-k can spend several
    slots on the same turn, so this selector first compresses every turn's
    scored chunks into one utility record, then selects the best representative
    chunk from each high-utility turn.  No gold labels are used.
    """
    if not scored_candidates:
        return [], {"selection_mode": "turn_utility", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    top_m = max(1, int(getattr(args, "turn_utility_top_m", 2)))
    use_prefilter_tiebreak = bool(
        int(getattr(args, "turn_utility_prefilter_tiebreak", 1))
    )

    head_vote_by_local = compute_candidate_head_vote_scores(
        per_head_scores,
        target_k,
    )

    by_turn = defaultdict(list)
    for local_idx, cand in enumerate(scored_candidates):
        by_turn[int(cand["turn_idx"])].append((int(local_idx), cand))

    turn_records = []
    for turn_idx, indexed_cands in by_turn.items():
        ranked_chunks = sorted(
            indexed_cands,
            key=lambda item: (
                -float(item[1].get("score", 0.0)),
                -float(head_vote_by_local.get(int(item[0]), 0.0)),
                -float(
                    candidate_prefilter_scores.get(
                        int(item[1]["candidate_id"]),
                        -1e9,
                    )
                ),
                int(item[1]["start_t"]),
                int(item[1]["candidate_id"]),
            ),
        )
        best_local_idx, best_candidate = ranked_chunks[0]
        top_scores = [
            float(cand.get("score", 0.0))
            for _, cand in ranked_chunks[: min(top_m, len(ranked_chunks))]
        ]
        positive_scores = [score for score in top_scores if score > 0.0]
        if positive_scores:
            utility = float(np.mean(positive_scores))
        else:
            utility = float(np.mean(top_scores)) if top_scores else 0.0
        best_head_vote = float(head_vote_by_local.get(int(best_local_idx), 0.0))
        best_prefilter_score = float(
            candidate_prefilter_scores.get(
                int(best_candidate["candidate_id"]),
                -1e9,
            )
        )
        turn_records.append(
            {
                "turn_idx": int(turn_idx),
                "best_local_idx": int(best_local_idx),
                "best_candidate_id": int(best_candidate["candidate_id"]),
                "start_t": int(best_candidate["start_t"]),
                "utility": float(utility),
                "best_qk": float(best_candidate.get("score", 0.0)),
                "best_head_vote": float(best_head_vote),
                "best_prefilter_score": float(best_prefilter_score),
                "num_scored_chunks": int(len(indexed_cands)),
            }
        )

    ranked_turns = sorted(
        turn_records,
        key=lambda item: (
            -float(item["utility"]),
            -float(item["best_qk"]),
            -float(item["best_head_vote"]),
            -float(item["best_prefilter_score"]) if use_prefilter_tiebreak else 0.0,
            int(item["start_t"]),
            int(item["turn_idx"]),
        ),
    )
    selected_turns = ranked_turns[:target_k]
    selected_local_ids = {int(item["best_local_idx"]) for item in selected_turns}
    turn_rank_by_candidate_id = {
        int(item["best_candidate_id"]): rank
        for rank, item in enumerate(selected_turns)
    }
    selected_candidates = [
        scored_candidates[local_idx]
        for local_idx in range(len(scored_candidates))
        if local_idx in selected_local_ids
    ]
    selected_candidates = sorted(
        selected_candidates,
        key=lambda cand: (
            int(turn_rank_by_candidate_id.get(int(cand["candidate_id"]), 10**9)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )

    debug = {
        "selection_mode": "turn_utility",
        "target_k": int(target_k),
        "turn_utility_top_m": int(top_m),
        "turn_utility_prefilter_tiebreak": bool(use_prefilter_tiebreak),
        "candidate_turns_before": int(len(turn_records)),
        "selected_turn_count": int(len(selected_turns)),
        "selected_duplicate_chunk_count": int(
            len(selected_candidates)
            - len({int(cand["turn_idx"]) for cand in selected_candidates})
        ),
        "turn_utility_preview": [
            {
                "rank": int(rank),
                "turn_idx": int(item["turn_idx"]),
                "best_candidate_id": int(item["best_candidate_id"]),
                "utility": float(item["utility"]),
                "best_qk": float(item["best_qk"]),
                "head_vote": float(item["best_head_vote"]),
                "prefilter_score": float(item["best_prefilter_score"]),
                "num_scored_chunks": int(item["num_scored_chunks"]),
            }
            for rank, item in enumerate(ranked_turns[:20], start=1)
        ],
        "turn_utility_ranked_turns": [
            {
                "rank": int(rank),
                "turn_idx": int(item["turn_idx"]),
                "best_candidate_id": int(item["best_candidate_id"]),
                "utility": float(item["utility"]),
                "best_qk": float(item["best_qk"]),
                "head_vote": float(item["best_head_vote"]),
                "prefilter_score": float(item["best_prefilter_score"]),
                "num_scored_chunks": int(item["num_scored_chunks"]),
            }
            for rank, item in enumerate(ranked_turns, start=1)
        ],
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "stage": "turn_utility",
                "score": float(cand.get("score", 0.0)),
            }
            for cand in selected_candidates[:30]
        ],
    }
    return selected_candidates, debug


def _rank_descending_numeric(values_by_key):
    finite_items = [
        (int(key), float(value))
        for key, value in values_by_key.items()
        if np.isfinite(float(value)) and float(value) > -1e8
    ]
    if not finite_items:
        return {}
    ranked = sorted(finite_items, key=lambda item: (-item[1], item[0]))
    ranks = {}
    last_value = None
    last_rank = 0
    for idx, (key, value) in enumerate(ranked, start=1):
        if last_value is None or value != last_value:
            last_rank = idx
            last_value = value
        ranks[int(key)] = int(last_rank)
    return ranks


def select_candidates_turn_rank_fusion(
    scored_candidates,
    per_head_scores,
    candidate_prefilter_scores,
    effective_route_top_k,
    args,
):
    """Select turns by rank-fusing Q-K, summary, and head-vote signals.

    This models the advisor-revised design more directly than a fixed weighted
    reranker: the request node receives lightweight block-summary scores, gets
    exact Q-K scores for the surviving blocks, and combines the available
    signal rankings without committing to hand-tuned weights.
    """
    if not scored_candidates:
        return [], {"selection_mode": "turn_rank_fusion", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    head_vote_by_local = compute_candidate_head_vote_scores(
        per_head_scores,
        target_k,
    )

    by_turn = defaultdict(list)
    qk_by_turn = {}
    prefilter_by_turn = {}
    node_summary_by_turn = {}
    head_vote_by_turn = {}
    for local_idx, cand in enumerate(scored_candidates):
        turn_idx = int(cand["turn_idx"])
        candidate_id = int(cand["candidate_id"])
        by_turn[turn_idx].append((int(local_idx), cand))
        qk_by_turn[turn_idx] = max(
            float(qk_by_turn.get(turn_idx, -1e9)),
            float(cand.get("score", 0.0)),
        )
        prefilter_by_turn[turn_idx] = max(
            float(prefilter_by_turn.get(turn_idx, -1e9)),
            float(candidate_prefilter_scores.get(candidate_id, -1e9)),
        )
        node_summary_by_turn[turn_idx] = max(
            float(node_summary_by_turn.get(turn_idx, -1e9)),
            float(cand.get("node_summary_score", -1e9)),
        )
        head_vote_by_turn[turn_idx] = max(
            float(head_vote_by_turn.get(turn_idx, 0.0)),
            float(head_vote_by_local.get(int(local_idx), 0.0)),
        )

    signal_ranks = {
        "qk": _rank_descending_numeric(qk_by_turn),
        "summary": _rank_descending_numeric(prefilter_by_turn),
        "node_summary": _rank_descending_numeric(node_summary_by_turn),
        "head_vote": _rank_descending_numeric(head_vote_by_turn),
    }
    active_signals = [
        name
        for name, ranks in signal_ranks.items()
        if ranks and (name != "head_vote" or any(value > 0.0 for value in head_vote_by_turn.values()))
    ]
    if not active_signals:
        active_signals = ["qk"]
        signal_ranks["qk"] = {
            int(turn_idx): rank
            for rank, turn_idx in enumerate(sorted(by_turn.keys()), start=1)
        }

    turn_count = max(1, len(by_turn))
    turn_records = []
    for turn_idx, indexed_cands in by_turn.items():
        ranked_chunks = sorted(
            indexed_cands,
            key=lambda item: (
                -float(item[1].get("score", 0.0)),
                -float(head_vote_by_local.get(int(item[0]), 0.0)),
                -float(
                    candidate_prefilter_scores.get(
                        int(item[1]["candidate_id"]),
                        -1e9,
                    )
                ),
                int(item[1]["start_t"]),
                int(item[1]["candidate_id"]),
            ),
        )
        best_local_idx, best_candidate = ranked_chunks[0]
        rank_values = [
            int(signal_ranks[name].get(int(turn_idx), turn_count + 1))
            for name in active_signals
        ]
        mean_rank = float(np.mean(rank_values)) if rank_values else float(turn_count + 1)
        mean_rank_pct = mean_rank / float(turn_count)
        fusion_score = 1.0 - min(1.0, mean_rank_pct)
        best_head_vote = float(head_vote_by_local.get(int(best_local_idx), 0.0))
        best_prefilter_score = float(
            candidate_prefilter_scores.get(
                int(best_candidate["candidate_id"]),
                -1e9,
            )
        )
        turn_records.append(
            {
                "turn_idx": int(turn_idx),
                "best_local_idx": int(best_local_idx),
                "best_candidate_id": int(best_candidate["candidate_id"]),
                "start_t": int(best_candidate["start_t"]),
                "utility": float(fusion_score),
                "rank_fusion_mean_rank": float(mean_rank),
                "rank_fusion_mean_rank_pct": float(mean_rank_pct),
                "rank_fusion_signal_ranks": {
                    name: int(signal_ranks[name].get(int(turn_idx), turn_count + 1))
                    for name in active_signals
                },
                "best_qk": float(best_candidate.get("score", 0.0)),
                "best_head_vote": float(best_head_vote),
                "best_prefilter_score": float(best_prefilter_score),
                "best_node_summary_score": float(
                    best_candidate.get("node_summary_score", -1e9)
                ),
                "num_scored_chunks": int(len(indexed_cands)),
            }
        )

    ranked_turns = sorted(
        turn_records,
        key=lambda item: (
            float(item["rank_fusion_mean_rank"]),
            -float(item["best_qk"]),
            -float(item["best_prefilter_score"]),
            -float(item["best_head_vote"]),
            int(item["start_t"]),
            int(item["turn_idx"]),
        ),
    )
    selected_turns = ranked_turns[:target_k]
    selected_local_ids = {int(item["best_local_idx"]) for item in selected_turns}
    turn_rank_by_candidate_id = {
        int(item["best_candidate_id"]): rank
        for rank, item in enumerate(selected_turns)
    }
    selected_candidates = [
        scored_candidates[local_idx]
        for local_idx in range(len(scored_candidates))
        if local_idx in selected_local_ids
    ]
    selected_candidates = sorted(
        selected_candidates,
        key=lambda cand: (
            int(turn_rank_by_candidate_id.get(int(cand["candidate_id"]), 10**9)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )

    def serialize_ranked_turn(item, rank):
        return {
            "rank": int(rank),
            "turn_idx": int(item["turn_idx"]),
            "best_candidate_id": int(item["best_candidate_id"]),
            "utility": float(item["utility"]),
            "rank_fusion_mean_rank": float(item["rank_fusion_mean_rank"]),
            "rank_fusion_mean_rank_pct": float(item["rank_fusion_mean_rank_pct"]),
            "rank_fusion_signal_ranks": item["rank_fusion_signal_ranks"],
            "best_qk": float(item["best_qk"]),
            "head_vote": float(item["best_head_vote"]),
            "prefilter_score": float(item["best_prefilter_score"]),
            "node_summary_score": float(item["best_node_summary_score"]),
            "num_scored_chunks": int(item["num_scored_chunks"]),
        }

    debug_ranked_turns = [
        serialize_ranked_turn(item, rank)
        for rank, item in enumerate(ranked_turns, start=1)
    ]
    debug = {
        "selection_mode": "turn_rank_fusion",
        "target_k": int(target_k),
        "candidate_turns_before": int(len(turn_records)),
        "selected_turn_count": int(len(selected_turns)),
        "selected_duplicate_chunk_count": int(
            len(selected_candidates)
            - len({int(cand["turn_idx"]) for cand in selected_candidates})
        ),
        "rank_fusion_active_signals": list(active_signals),
        "turn_rank_fusion_preview": debug_ranked_turns[:20],
        # Reuse the existing diagnostic channel so survival/oracle summaries
        # compare this selector to turn_utility without another output schema.
        "turn_utility_preview": debug_ranked_turns[:20],
        "turn_utility_ranked_turns": debug_ranked_turns,
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "stage": "turn_rank_fusion",
                "score": float(cand.get("score", 0.0)),
            }
            for cand in selected_candidates[:30]
        ],
    }
    return selected_candidates, debug


def _candidate_primary_selected_topic(candidate, selected_topic_rank):
    topic_ids = [
        int(topic_id)
        for topic_id in candidate.get("topic_ids", [])
        if int(topic_id) in selected_topic_rank
    ]
    if not topic_ids:
        return None
    return sorted(topic_ids, key=lambda tid: selected_topic_rank.get(tid, 10**9))[0]


def select_candidates_topic_balanced(
    scored_candidates,
    per_head_scores,
    effective_route_top_k,
    use_per_head,
    selected_topic_ids,
    args,
):
    """Spend part of the final evidence budget on each rescued topic.

    Adaptive topic rescue can correctly keep a second possible topic, but plain
    global top-k may still let the dominant topic consume the whole final budget.
    This selector first reserves a small topic-local quota, then backfills by the
    usual global Q-K ranking.
    """
    if not scored_candidates:
        return [], {"selection_mode": "topic_balanced", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    selected_topic_ids = [int(tid) for tid in selected_topic_ids]
    selected_topic_rank = {int(tid): idx for idx, tid in enumerate(selected_topic_ids)}

    if len(selected_topic_ids) <= 1:
        selected, debug = select_candidates_chunk_topk(
            scored_candidates,
            per_head_scores,
            target_k,
            use_per_head,
        )
        debug = dict(debug)
        debug["selection_mode"] = "topic_balanced_fallback_single_topic"
        debug["topic_balanced_topic_count"] = int(len(selected_topic_ids))
        return selected, debug

    qk_core, qk_core_debug = select_candidates_chunk_topk(
        scored_candidates,
        per_head_scores,
        target_k,
        use_per_head,
    )
    qk_core_ids = {int(cand["candidate_id"]) for cand in qk_core}
    ranked = sorted(
        scored_candidates,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            selected_topic_rank.get(
                _candidate_primary_selected_topic(cand, selected_topic_rank),
                10**9,
            ),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )

    by_topic = defaultdict(list)
    for cand in ranked:
        topic_id = _candidate_primary_selected_topic(cand, selected_topic_rank)
        if topic_id is not None:
            by_topic[int(topic_id)].append(cand)

    topic_count = len([tid for tid in selected_topic_ids if by_topic.get(int(tid))])
    if topic_count <= 1:
        selected = ranked[:target_k]
        return selected, {
            "selection_mode": "topic_balanced_fallback_one_nonempty_topic",
            "target_k": int(target_k),
            "topic_balanced_topic_count": int(topic_count),
            "qk_core_debug": qk_core_debug,
        }

    min_per_topic = max(
        1,
        int(getattr(args, "topic_balanced_min_per_topic", 2)),
    )
    max_reserved = max(
        1,
        int(np.floor(float(target_k) * 0.6)),
    )
    min_per_topic = min(min_per_topic, max(1, max_reserved // topic_count))

    selected = []
    selected_ids = set()
    selected_stage = {}
    topic_quota_added = defaultdict(int)

    def add_candidate(cand, stage):
        candidate_id = int(cand["candidate_id"])
        if candidate_id in selected_ids or len(selected) >= target_k:
            return False
        selected.append(cand)
        selected_ids.add(candidate_id)
        selected_stage[candidate_id] = stage
        topic_id = _candidate_primary_selected_topic(cand, selected_topic_rank)
        if topic_id is not None:
            topic_quota_added[int(topic_id)] += 1
        return True

    # First pass: give each selected/rescued topic a small local budget.
    for topic_id in selected_topic_ids:
        added_for_topic = 0
        for cand in by_topic.get(int(topic_id), []):
            if added_for_topic >= min_per_topic:
                break
            if add_candidate(cand, "topic_quota"):
                added_for_topic += 1

    # Second pass: keep the usual global high-QK anchors, then pure backfill.
    for cand in qk_core:
        if len(selected) >= target_k:
            break
        add_candidate(cand, "qk_core_backfill")

    for cand in ranked:
        if len(selected) >= target_k:
            break
        add_candidate(cand, "global_backfill")

    selected = sorted(
        selected,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )
    selected_topic_counts = defaultdict(int)
    for cand in selected:
        topic_id = _candidate_primary_selected_topic(cand, selected_topic_rank)
        if topic_id is not None:
            selected_topic_counts[int(topic_id)] += 1

    debug = {
        "selection_mode": "topic_balanced",
        "target_k": int(target_k),
        "topic_balanced_topic_count": int(topic_count),
        "topic_balanced_min_per_topic": int(min_per_topic),
        "topic_balanced_selected_topics": [int(tid) for tid in selected_topic_ids],
        "topic_balanced_quota_added": {
            int(tid): int(topic_quota_added.get(int(tid), 0))
            for tid in selected_topic_ids
        },
        "topic_balanced_selected_topic_counts": {
            int(tid): int(selected_topic_counts.get(int(tid), 0))
            for tid in selected_topic_ids
        },
        "qk_core_overlap": int(
            len(qk_core_ids & {int(cand["candidate_id"]) for cand in selected})
        ),
        "qk_core_debug": qk_core_debug,
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "topic_id": _candidate_primary_selected_topic(cand, selected_topic_rank),
                "stage": selected_stage.get(int(cand["candidate_id"]), ""),
                "score": float(cand.get("score", 0.0)),
            }
            for cand in selected[:30]
        ],
    }
    return selected, debug


def select_candidates_topic_soft_rescue(
    scored_candidates,
    per_head_scores,
    effective_route_top_k,
    use_per_head,
    selected_topic_ids,
    args,
):
    """Start from global Q-K top-k, then softly rescue missing topics.

    This is a less brittle version of topic_balanced.  It keeps the current
    global Q-K selection as the default, and only replaces weak selected chunks
    when a rescued topic is missing and its best candidate is close enough to
    the replacement boundary.
    """
    if not scored_candidates:
        return [], {"selection_mode": "topic_soft_rescue", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    selected_topic_ids = [int(tid) for tid in selected_topic_ids]
    selected_topic_rank = {int(tid): idx for idx, tid in enumerate(selected_topic_ids)}

    selected, qk_core_debug = select_candidates_chunk_topk(
        scored_candidates,
        per_head_scores,
        target_k,
        use_per_head,
    )
    selected = sorted(
        selected,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )[:target_k]

    if len(selected_topic_ids) <= 1 or not selected:
        debug = dict(qk_core_debug)
        debug.update(
            {
                "selection_mode": "topic_soft_rescue_fallback",
                "topic_soft_rescue_topic_count": int(len(selected_topic_ids)),
                "topic_soft_rescue_replacements": 0,
            }
        )
        return selected, debug

    ranked = sorted(
        scored_candidates,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            selected_topic_rank.get(
                _candidate_primary_selected_topic(cand, selected_topic_rank),
                10**9,
            ),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )

    by_topic = defaultdict(list)
    for cand in ranked:
        topic_id = _candidate_primary_selected_topic(cand, selected_topic_rank)
        if topic_id is not None:
            by_topic[int(topic_id)].append(cand)

    selected_ids = {int(cand["candidate_id"]) for cand in selected}
    selected_stage = {int(cand["candidate_id"]): "qk_core" for cand in selected}

    selected_topic_counts = defaultdict(int)
    for cand in selected:
        topic_id = _candidate_primary_selected_topic(cand, selected_topic_rank)
        if topic_id is not None:
            selected_topic_counts[int(topic_id)] += 1

    selected_scores = [float(cand.get("score", 0.0)) for cand in selected]
    score_span = max(selected_scores) - min(selected_scores) if selected_scores else 0.0
    margin_ratio = max(
        0.0,
        float(getattr(args, "topic_soft_rescue_margin_ratio", 0.15)),
    )
    min_score_ratio = max(
        0.0,
        float(getattr(args, "topic_soft_rescue_min_score_ratio", 0.90)),
    )
    max_replacements = max(
        0,
        int(getattr(args, "topic_soft_rescue_max_replacements", 2)),
    )
    score_tolerance = max(0.0, float(score_span) * margin_ratio)

    replacements = []
    skipped_low_score = []

    def find_drop_index(candidate):
        candidate_topic = _candidate_primary_selected_topic(candidate, selected_topic_rank)
        eligible = []
        for idx, cand in enumerate(selected):
            topic_id = _candidate_primary_selected_topic(cand, selected_topic_rank)
            # Do not remove the only selected evidence for another rescued topic.
            if (
                topic_id is not None
                and topic_id != candidate_topic
                and selected_topic_counts.get(int(topic_id), 0) <= 1
            ):
                continue
            eligible.append(
                (
                    float(cand.get("score", 0.0)),
                    -selected_topic_rank.get(topic_id, 10**9),
                    -idx,
                    idx,
                )
            )
        if not eligible:
            return None
        return sorted(eligible)[0][3]

    for topic_id in selected_topic_ids:
        if len(replacements) >= max_replacements:
            break
        topic_id = int(topic_id)
        if selected_topic_counts.get(topic_id, 0) > 0:
            continue

        replacement_candidate = None
        for cand in by_topic.get(topic_id, []):
            if int(cand["candidate_id"]) not in selected_ids:
                replacement_candidate = cand
                break
        if replacement_candidate is None:
            continue

        drop_idx = find_drop_index(replacement_candidate)
        if drop_idx is None:
            continue

        dropped = selected[drop_idx]
        candidate_score = float(replacement_candidate.get("score", 0.0))
        dropped_score = float(dropped.get("score", 0.0))
        score_gap = float(dropped_score - candidate_score)
        min_allowed_score = dropped_score * min_score_ratio
        if candidate_score < min_allowed_score and score_gap > score_tolerance:
            skipped_low_score.append(
                {
                    "topic_id": int(topic_id),
                    "candidate_id": int(replacement_candidate["candidate_id"]),
                    "candidate_score": candidate_score,
                    "dropped_candidate_id": int(dropped["candidate_id"]),
                    "dropped_score": dropped_score,
                    "score_gap": score_gap,
                    "score_tolerance": float(score_tolerance),
                    "min_allowed_score": float(min_allowed_score),
                }
            )
            continue

        dropped_topic = _candidate_primary_selected_topic(dropped, selected_topic_rank)
        selected_ids.remove(int(dropped["candidate_id"]))
        selected_ids.add(int(replacement_candidate["candidate_id"]))
        if dropped_topic is not None:
            selected_topic_counts[int(dropped_topic)] -= 1
        selected_topic_counts[int(topic_id)] += 1
        selected[drop_idx] = replacement_candidate
        selected_stage[int(replacement_candidate["candidate_id"])] = "topic_soft_rescue"
        replacements.append(
            {
                "topic_id": int(topic_id),
                "candidate_id": int(replacement_candidate["candidate_id"]),
                "candidate_score": candidate_score,
                "dropped_candidate_id": int(dropped["candidate_id"]),
                "dropped_topic_id": int(dropped_topic) if dropped_topic is not None else -1,
                "dropped_score": dropped_score,
                "score_gap": score_gap,
                "score_tolerance": float(score_tolerance),
                "min_allowed_score": float(min_allowed_score),
            }
        )

    selected = sorted(
        selected,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )

    final_topic_counts = defaultdict(int)
    for cand in selected:
        topic_id = _candidate_primary_selected_topic(cand, selected_topic_rank)
        if topic_id is not None:
            final_topic_counts[int(topic_id)] += 1

    debug = {
        "selection_mode": "topic_soft_rescue",
        "target_k": int(target_k),
        "topic_soft_rescue_topic_count": int(len(selected_topic_ids)),
        "topic_soft_rescue_selected_topics": [int(tid) for tid in selected_topic_ids],
        "topic_soft_rescue_replacements": int(len(replacements)),
        "topic_soft_rescue_max_replacements": int(max_replacements),
        "topic_soft_rescue_margin_ratio": float(margin_ratio),
        "topic_soft_rescue_min_score_ratio": float(min_score_ratio),
        "topic_soft_rescue_score_tolerance": float(score_tolerance),
        "topic_soft_rescue_selected_topic_counts": {
            int(tid): int(final_topic_counts.get(int(tid), 0))
            for tid in selected_topic_ids
        },
        "topic_soft_rescue_replacement_details": replacements[:20],
        "topic_soft_rescue_skipped_low_score": skipped_low_score[:20],
        "qk_core_debug": qk_core_debug,
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "topic_id": _candidate_primary_selected_topic(cand, selected_topic_rank),
                "stage": selected_stage.get(int(cand["candidate_id"]), "qk_core"),
                "score": float(cand.get("score", 0.0)),
            }
            for cand in selected[:30]
        ],
    }
    return selected, debug


def select_candidates_turn_unique(
    scored_candidates,
    per_head_scores,
    effective_route_top_k,
    use_per_head,
    args,
):
    """Prefer one strong chunk per turn, then backfill by raw Q-K.

    The default answer path consumes whole turns.  If several high-QK chunks
    come from the same turn, they are often redundant for answer generation but
    still count as transferred KV.  This selector keeps the raw-QK anchors, then
    spends the remaining budget on distinct turns before any duplicate backfill.
    """
    if not scored_candidates:
        return [], {"selection_mode": "turn_unique", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    anchor_count = max(
        0,
        int(getattr(args, "route_pack_anchor_count", 3)),
    )
    anchor_count = min(anchor_count, target_k)

    qk_core, qk_core_debug = select_candidates_chunk_topk(
        scored_candidates,
        per_head_scores,
        target_k,
        use_per_head,
    )
    qk_core_ranked = sorted(
        qk_core,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )[:target_k]
    qk_ranked = sorted(
        scored_candidates,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )

    selected = []
    selected_ids = set()
    selected_stage = {}
    covered_turns = set()

    def add_candidate(cand, stage):
        candidate_id = int(cand["candidate_id"])
        if candidate_id in selected_ids:
            return False
        if len(selected) >= target_k:
            return False
        selected.append(cand)
        selected_ids.add(candidate_id)
        selected_stage[candidate_id] = stage
        covered_turns.add(int(cand["turn_idx"]))
        return True

    for cand in qk_core_ranked:
        if len(selected) >= anchor_count:
            break
        if int(cand["turn_idx"]) in covered_turns:
            continue
        add_candidate(cand, "anchor")

    unique_turn_added = 0
    duplicate_deferred = 0
    for cand in qk_core_ranked:
        if len(selected) >= target_k:
            break
        turn_idx = int(cand["turn_idx"])
        candidate_id = int(cand["candidate_id"])
        if candidate_id in selected_ids:
            continue
        if turn_idx in covered_turns:
            duplicate_deferred += 1
            continue
        if add_candidate(cand, "unique_turn"):
            unique_turn_added += 1

    global_unique_turn_added = 0
    for cand in qk_ranked:
        if len(selected) >= target_k:
            break
        turn_idx = int(cand["turn_idx"])
        candidate_id = int(cand["candidate_id"])
        if candidate_id in selected_ids:
            continue
        if turn_idx in covered_turns:
            continue
        if add_candidate(cand, "global_unique_turn"):
            global_unique_turn_added += 1

    duplicate_backfill_added = 0
    for cand in qk_ranked:
        if len(selected) >= target_k:
            break
        candidate_id = int(cand["candidate_id"])
        if candidate_id in selected_ids:
            continue
        if add_candidate(cand, "duplicate_backfill"):
            duplicate_backfill_added += 1

    # Keep a few high-salience anchors up front, then make the rest readable as
    # chronological meeting evidence.
    anchor_order = [
        int(cand["candidate_id"])
        for cand in sorted(
            [
                cand
                for cand in selected
                if selected_stage.get(int(cand["candidate_id"])) == "anchor"
            ],
            key=lambda cand: -float(cand.get("score", 0.0)),
        )
    ]
    anchor_order_set = set(anchor_order)
    tail = [
        cand for cand in selected
        if int(cand["candidate_id"]) not in anchor_order_set
    ]
    ordered_selected = [
        next(cand for cand in selected if int(cand["candidate_id"]) == candidate_id)
        for candidate_id in anchor_order
    ] + sorted(
        tail,
        key=lambda cand: (
            int(cand["start_t"]),
            int(cand["turn_idx"]),
            -float(cand.get("score", 0.0)),
        ),
    )

    selected_turn_count = len({int(cand["turn_idx"]) for cand in ordered_selected})
    debug = {
        "selection_mode": "turn_unique",
        "target_k": int(target_k),
        "anchor_count": int(
            len(
                [
                    stage
                    for stage in selected_stage.values()
                    if stage == "anchor"
                ]
            )
        ),
        "anchor_target_count": int(anchor_count),
        "unique_turn_added": int(unique_turn_added),
        "global_unique_turn_added": int(global_unique_turn_added),
        "duplicate_deferred": int(duplicate_deferred),
        "duplicate_backfill_added": int(duplicate_backfill_added),
        "selected_turn_count": int(selected_turn_count),
        "selected_duplicate_chunk_count": int(len(ordered_selected) - selected_turn_count),
        "qk_core_debug": qk_core_debug,
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "stage": selected_stage.get(int(cand["candidate_id"]), ""),
                "score": float(cand.get("score", 0.0)),
            }
            for cand in ordered_selected[:30]
        ],
    }
    return ordered_selected, debug


def select_candidates_turn_unique_guard(
    scored_candidates,
    per_head_scores,
    effective_route_top_k,
    use_per_head,
    args,
):
    """Conservative turn de-duplication over the current Q-K core.

    Start from the same chunk set as the current selector.  Only replace a
    duplicate-turn tail chunk when a not-yet-covered turn has a similar Q-K
    score.  This tests coverage without throwing away strong repeated evidence.
    """
    if not scored_candidates:
        return [], {"selection_mode": "turn_unique_guard", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    anchor_count = max(0, int(getattr(args, "route_pack_anchor_count", 4)))
    anchor_count = min(anchor_count, target_k)
    replacement_budget = max(
        0,
        int(getattr(args, "turn_unique_max_replacements", 2)),
    )
    replacement_budget = min(replacement_budget, max(0, target_k - anchor_count))
    replacement_min_ratio = max(
        0.0,
        float(getattr(args, "turn_unique_replacement_min_ratio", 0.85)),
    )

    qk_core, qk_core_debug = select_candidates_chunk_topk(
        scored_candidates,
        per_head_scores,
        target_k,
        use_per_head,
    )
    selected = sorted(
        qk_core,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )[:target_k]
    qk_ranked = sorted(
        scored_candidates,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )

    selected_ids = {int(cand["candidate_id"]) for cand in selected}
    selected_stage = {int(cand["candidate_id"]): "qk_core" for cand in selected}
    anchor_ids = {
        int(cand["candidate_id"])
        for cand in sorted(
            selected,
            key=lambda cand: (
                -float(cand.get("score", 0.0)),
                int(cand["start_t"]),
                int(cand["candidate_id"]),
            ),
        )[:anchor_count]
    }
    turn_counts = defaultdict(int)
    for cand in selected:
        turn_counts[int(cand["turn_idx"])] += 1

    initial_duplicate_chunk_count = sum(
        max(0, int(count) - 1) for count in turn_counts.values()
    )
    replacement_added = 0
    replacement_skipped_no_drop = 0
    replacement_skipped_low_score = 0
    dropped_candidates = []

    def find_drop_index_for_candidate(candidate):
        eligible = []
        for idx, cand in enumerate(selected):
            candidate_id = int(cand["candidate_id"])
            if candidate_id in anchor_ids:
                continue
            turn_idx = int(cand["turn_idx"])
            if turn_counts.get(turn_idx, 0) <= 1:
                continue
            eligible.append(
                (
                    float(cand.get("score", 0.0)),
                    -idx,
                    idx,
                )
            )
        if not eligible:
            return None
        return sorted(eligible)[0][2]

    for candidate in qk_ranked:
        if replacement_added >= replacement_budget:
            break
        candidate_id = int(candidate["candidate_id"])
        candidate_turn = int(candidate["turn_idx"])
        if candidate_id in selected_ids:
            continue
        if turn_counts.get(candidate_turn, 0) > 0:
            continue

        drop_idx = find_drop_index_for_candidate(candidate)
        if drop_idx is None:
            replacement_skipped_no_drop += 1
            break

        dropped = selected[drop_idx]
        dropped_score = float(dropped.get("score", 0.0))
        candidate_score = float(candidate.get("score", 0.0))
        required_score = dropped_score * replacement_min_ratio
        if candidate_score < required_score:
            replacement_skipped_low_score += 1
            break

        selected[drop_idx] = candidate
        selected_ids.remove(int(dropped["candidate_id"]))
        selected_ids.add(candidate_id)
        turn_counts[int(dropped["turn_idx"])] -= 1
        turn_counts[candidate_turn] += 1
        selected_stage[candidate_id] = "guard_unique_turn"
        replacement_added += 1
        dropped_candidates.append(
            {
                "candidate_id": int(dropped["candidate_id"]),
                "turn_idx": int(dropped["turn_idx"]),
                "score": dropped_score,
                "replacement_id": candidate_id,
                "replacement_turn_idx": candidate_turn,
                "replacement_score": candidate_score,
                "required_score": float(required_score),
            }
        )

    ordered_selected = sorted(
        selected,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )
    selected_turn_count = len({int(cand["turn_idx"]) for cand in ordered_selected})
    final_duplicate_chunk_count = len(ordered_selected) - selected_turn_count

    debug = {
        "selection_mode": "turn_unique_guard",
        "target_k": int(target_k),
        "anchor_count": int(len(anchor_ids)),
        "anchor_target_count": int(anchor_count),
        "replacement_budget": int(replacement_budget),
        "replacement_added": int(replacement_added),
        "replacement_skipped_no_drop": int(replacement_skipped_no_drop),
        "replacement_skipped_low_score": int(replacement_skipped_low_score),
        "replacement_min_ratio": float(replacement_min_ratio),
        "unique_turn_added": 0,
        "global_unique_turn_added": int(replacement_added),
        "duplicate_deferred": int(initial_duplicate_chunk_count),
        "duplicate_backfill_added": int(final_duplicate_chunk_count),
        "selected_turn_count": int(selected_turn_count),
        "selected_duplicate_chunk_count": int(final_duplicate_chunk_count),
        "initial_duplicate_chunk_count": int(initial_duplicate_chunk_count),
        "dropped_candidates": dropped_candidates[:20],
        "qk_core_debug": qk_core_debug,
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "stage": selected_stage.get(int(cand["candidate_id"]), "qk_core"),
                "score": float(cand.get("score", 0.0)),
            }
            for cand in ordered_selected[:30]
        ],
    }
    return ordered_selected, debug


def select_candidates_turn_unique_soft(
    scored_candidates,
    per_head_scores,
    effective_route_top_k,
    use_per_head,
    args,
):
    """Replace duplicate-turn tail chunks only inside the Q-K boundary band.

    The answer path consumes whole turns, so repeated chunks from one turn can
    waste transfer budget.  Unlike ``turn_unique``, this selector starts from
    the current Q-K top-k set and only swaps weak duplicate-turn tail chunks
    when the best uncovered-turn candidate is close enough to the selected
    boundary for the Q-K ordering to be treated as uncertain.
    """
    if not scored_candidates:
        return [], {"selection_mode": "turn_unique_soft", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    anchor_count = max(0, int(getattr(args, "route_pack_anchor_count", 4)))
    anchor_count = min(anchor_count, target_k)
    replacement_budget = max(
        0,
        int(getattr(args, "turn_unique_max_replacements", 1)),
    )
    replacement_budget = min(replacement_budget, max(0, target_k - anchor_count))
    soft_margin_ratio = max(
        0.0,
        float(getattr(args, "turn_unique_soft_margin_ratio", 0.15)),
    )

    qk_core, qk_core_debug = select_candidates_chunk_topk(
        scored_candidates,
        per_head_scores,
        target_k,
        use_per_head,
    )
    selected = sorted(
        qk_core,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )[:target_k]
    qk_ranked = sorted(
        scored_candidates,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            -float(cand.get("prefilter_score", -1e9)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )

    selected_ids = {int(cand["candidate_id"]) for cand in selected}
    selected_stage = {int(cand["candidate_id"]): "qk_core" for cand in selected}
    anchor_ids = {
        int(cand["candidate_id"])
        for cand in sorted(
            selected,
            key=lambda cand: (
                -float(cand.get("score", 0.0)),
                int(cand["start_t"]),
                int(cand["candidate_id"]),
            ),
        )[:anchor_count]
    }
    turn_counts = defaultdict(int)
    for cand in selected:
        turn_counts[int(cand["turn_idx"])] += 1

    initial_duplicate_chunk_count = sum(
        max(0, int(count) - 1) for count in turn_counts.values()
    )
    selected_scores = [float(cand.get("score", 0.0)) for cand in selected]
    if selected_scores:
        top_score = max(selected_scores)
        cutoff_score = min(selected_scores)
        selected_score_span = max(0.0, float(top_score - cutoff_score))
    else:
        cutoff_score = 0.0
        selected_score_span = 0.0

    window_scores = [
        float(cand.get("score", 0.0))
        for cand in qk_ranked[: min(len(qk_ranked), max(target_k * 2, target_k + 1))]
    ]
    local_gaps = [
        max(0.0, float(window_scores[idx] - window_scores[idx + 1]))
        for idx in range(len(window_scores) - 1)
    ]
    positive_gaps = [gap for gap in local_gaps if gap > 0.0]
    median_gap = float(np.median(positive_gaps)) if positive_gaps else 0.0
    score_tolerance = max(selected_score_span * soft_margin_ratio, median_gap)

    replacement_added = 0
    replacement_skipped_no_drop = 0
    replacement_skipped_low_score = 0
    dropped_candidates = []
    low_score_candidates = []

    def find_drop_index_for_candidate(candidate):
        eligible = []
        for idx, cand in enumerate(selected):
            candidate_id = int(cand["candidate_id"])
            if candidate_id in anchor_ids:
                continue
            turn_idx = int(cand["turn_idx"])
            if turn_counts.get(turn_idx, 0) <= 1:
                continue
            eligible.append(
                (
                    float(cand.get("score", 0.0)),
                    float(cand.get("prefilter_score", -1e9)),
                    -idx,
                    idx,
                )
            )
        if not eligible:
            return None
        return sorted(eligible)[0][3]

    for candidate in qk_ranked:
        if replacement_added >= replacement_budget:
            break
        candidate_id = int(candidate["candidate_id"])
        candidate_turn = int(candidate["turn_idx"])
        if candidate_id in selected_ids:
            continue
        if turn_counts.get(candidate_turn, 0) > 0:
            continue

        drop_idx = find_drop_index_for_candidate(candidate)
        if drop_idx is None:
            replacement_skipped_no_drop += 1
            break

        dropped = selected[drop_idx]
        dropped_score = float(dropped.get("score", 0.0))
        candidate_score = float(candidate.get("score", 0.0))
        score_gap = float(dropped_score - candidate_score)
        if score_gap > score_tolerance:
            replacement_skipped_low_score += 1
            low_score_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "turn_idx": candidate_turn,
                    "score": candidate_score,
                    "dropped_candidate_id": int(dropped["candidate_id"]),
                    "dropped_score": dropped_score,
                    "score_gap": score_gap,
                    "score_tolerance": float(score_tolerance),
                    "cutoff_score": float(cutoff_score),
                }
            )
            break

        selected[drop_idx] = candidate
        selected_ids.remove(int(dropped["candidate_id"]))
        selected_ids.add(candidate_id)
        turn_counts[int(dropped["turn_idx"])] -= 1
        turn_counts[candidate_turn] += 1
        selected_stage[candidate_id] = "soft_unique_turn"
        replacement_added += 1
        dropped_candidates.append(
            {
                "candidate_id": int(dropped["candidate_id"]),
                "turn_idx": int(dropped["turn_idx"]),
                "score": dropped_score,
                "replacement_id": candidate_id,
                "replacement_turn_idx": candidate_turn,
                "replacement_score": candidate_score,
                "score_gap": score_gap,
                "score_tolerance": float(score_tolerance),
            }
        )

    ordered_selected = sorted(
        selected,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )
    selected_turn_count = len({int(cand["turn_idx"]) for cand in ordered_selected})
    final_duplicate_chunk_count = len(ordered_selected) - selected_turn_count

    debug = {
        "selection_mode": "turn_unique_soft",
        "target_k": int(target_k),
        "anchor_count": int(len(anchor_ids)),
        "anchor_target_count": int(anchor_count),
        "replacement_budget": int(replacement_budget),
        "replacement_added": int(replacement_added),
        "replacement_skipped_no_drop": int(replacement_skipped_no_drop),
        "replacement_skipped_low_score": int(replacement_skipped_low_score),
        "replacement_score_tolerance": float(score_tolerance),
        "replacement_score_span": float(selected_score_span),
        "replacement_median_gap": float(median_gap),
        "replacement_soft_margin_ratio": float(soft_margin_ratio),
        "unique_turn_added": 0,
        "global_unique_turn_added": int(replacement_added),
        "duplicate_deferred": int(initial_duplicate_chunk_count),
        "duplicate_backfill_added": int(final_duplicate_chunk_count),
        "selected_turn_count": int(selected_turn_count),
        "selected_duplicate_chunk_count": int(final_duplicate_chunk_count),
        "initial_duplicate_chunk_count": int(initial_duplicate_chunk_count),
        "dropped_candidates": dropped_candidates[:20],
        "low_score_candidates": low_score_candidates[:20],
        "qk_core_debug": qk_core_debug,
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "stage": selected_stage.get(int(cand["candidate_id"]), "qk_core"),
                "score": float(cand.get("score", 0.0)),
            }
            for cand in ordered_selected[:30]
        ],
    }
    return ordered_selected, debug


def select_candidates_turn_rerank(
    scored_candidates,
    per_head_scores,
    candidate_prefilter_scores,
    transcripts,
    query_text,
    effective_route_top_k,
    args,
):
    if not scored_candidates:
        return [], {"selection_mode": "turn_rerank", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    query_tokens = set(tokenize_lexical_text(query_text))

    head_vote_by_local = compute_candidate_head_vote_scores(
        per_head_scores,
        target_k,
    )
    qk_by_turn = {}
    prefilter_by_turn = {}
    by_turn = defaultdict(list)
    for local_idx, cand in enumerate(scored_candidates):
        turn_idx = int(cand["turn_idx"])
        by_turn[turn_idx].append((local_idx, cand))
        qk_by_turn[turn_idx] = max(
            float(qk_by_turn.get(turn_idx, -1e9)),
            float(cand.get("score", 0.0)),
        )
        prefilter_by_turn[turn_idx] = max(
            float(prefilter_by_turn.get(turn_idx, -1e9)),
            float(candidate_prefilter_scores.get(int(cand["candidate_id"]), -1e9)),
        )

    normalized_qk = normalize_numeric_map(qk_by_turn)
    normalized_prefilter = normalize_numeric_map(prefilter_by_turn)
    qk_weight = float(getattr(args, "turn_rerank_qk_weight", 0.65))
    lexical_weight = float(getattr(args, "turn_rerank_lexical_weight", 0.25))
    head_vote_weight = float(getattr(args, "turn_rerank_head_vote_weight", 0.10))

    turn_records = []
    for turn_idx, indexed_cands in by_turn.items():
        speaker, turn_text = get_turn_text(transcripts, turn_idx)
        turn_tokens = set(tokenize_lexical_text(turn_text))
        lexical_overlap = 0.0
        if query_tokens:
            lexical_overlap = float(len(query_tokens & turn_tokens)) / float(
                len(query_tokens)
            )
        max_head_vote = max(
            float(head_vote_by_local.get(int(local_idx), 0.0))
            for local_idx, _ in indexed_cands
        )
        best_local_idx, best_candidate = sorted(
            indexed_cands,
            key=lambda item: (
                -float(item[1].get("score", 0.0)),
                -float(head_vote_by_local.get(int(item[0]), 0.0)),
                int(item[1]["start_t"]),
                int(item[1]["candidate_id"]),
            ),
        )[0]
        turn_score = (
            qk_weight * float(normalized_qk.get(int(turn_idx), 0.0))
            + lexical_weight * float(lexical_overlap)
            + head_vote_weight * float(max_head_vote)
        )
        turn_records.append(
            {
                "turn_idx": int(turn_idx),
                "speaker": speaker,
                "start_t": int(best_candidate["start_t"]),
                "best_candidate_id": int(best_candidate["candidate_id"]),
                "best_local_idx": int(best_local_idx),
                "best_qk": float(best_candidate.get("score", 0.0)),
                "norm_qk": float(normalized_qk.get(int(turn_idx), 0.0)),
                "lexical_overlap": float(lexical_overlap),
                "head_vote": float(max_head_vote),
                "prefilter_score_norm": float(
                    normalized_prefilter.get(int(turn_idx), 0.0)
                ),
                "turn_score": float(turn_score),
                "num_scored_chunks": int(len(indexed_cands)),
            }
        )

    ranked_turns = sorted(
        turn_records,
        key=lambda item: (
            -float(item["turn_score"]),
            int(item["start_t"]),
            int(item["turn_idx"]),
        ),
    )
    selected_turns = ranked_turns[:target_k]
    selected_local_ids = {int(item["best_local_idx"]) for item in selected_turns}
    selected_candidates = [
        scored_candidates[local_idx]
        for local_idx in range(len(scored_candidates))
        if local_idx in selected_local_ids
    ]
    selected_candidates = sorted(
        selected_candidates,
        key=lambda cand: (
            next(
                idx
                for idx, item in enumerate(selected_turns)
                if int(item["best_candidate_id"]) == int(cand["candidate_id"])
            ),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )

    debug = {
        "selection_mode": "turn_rerank",
        "target_k": int(target_k),
        "candidate_turns_before": int(len(turn_records)),
        "selected_turn_count": int(len(selected_turns)),
        "turn_rerank_qk_weight": float(qk_weight),
        "turn_rerank_lexical_weight": float(lexical_weight),
        "turn_rerank_head_vote_weight": float(head_vote_weight),
        "turn_rerank_preview": [
            {
                "turn_idx": int(item["turn_idx"]),
                "speaker": item["speaker"],
                "best_candidate_id": int(item["best_candidate_id"]),
                "turn_score": float(item["turn_score"]),
                "best_qk": float(item["best_qk"]),
                "norm_qk": float(item["norm_qk"]),
                "lexical_overlap": float(item["lexical_overlap"]),
                "head_vote": float(item["head_vote"]),
                "num_scored_chunks": int(item["num_scored_chunks"]),
            }
            for item in ranked_turns[:20]
        ],
    }
    return selected_candidates, debug


def select_candidates_hybrid(
    scored_candidates,
    per_head_scores,
    candidate_prefilter_scores,
    transcripts,
    query_text,
    effective_route_top_k,
    args,
):
    """Keep a raw-QK core, then add turn-level coverage candidates."""
    if not scored_candidates:
        return [], {"selection_mode": "hybrid", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    core_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "route_hybrid_core_ratio", 0.5))),
    )
    core_count = int(np.ceil(float(target_k) * core_ratio))
    core_count = min(target_k, max(1, core_count))
    core_max_per_turn = max(
        1,
        int(getattr(args, "route_hybrid_core_max_per_turn", 1)),
    )

    qk_ranked = sorted(
        scored_candidates,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )
    core_candidates = qk_ranked[:core_count]

    selected = []
    selected_ids = set()
    covered_turns = set()
    core_turn_counts = defaultdict(int)
    selected_stage = {}
    for cand in qk_ranked:
        if len(selected) >= core_count:
            break
        candidate_id = int(cand["candidate_id"])
        turn_idx = int(cand["turn_idx"])
        if candidate_id in selected_ids:
            continue
        if core_turn_counts[turn_idx] >= core_max_per_turn:
            continue
        selected.append(cand)
        selected_ids.add(candidate_id)
        covered_turns.add(turn_idx)
        core_turn_counts[turn_idx] += 1
        selected_stage[candidate_id] = "qk_core"

    turn_candidates, turn_debug = select_candidates_turn_rerank(
        scored_candidates,
        per_head_scores,
        candidate_prefilter_scores,
        transcripts,
        query_text,
        len(scored_candidates),
        args,
    )
    coverage_added = 0
    for cand in turn_candidates:
        if len(selected) >= target_k:
            break
        candidate_id = int(cand["candidate_id"])
        turn_idx = int(cand["turn_idx"])
        if candidate_id in selected_ids or turn_idx in covered_turns:
            continue
        selected.append(cand)
        selected_ids.add(candidate_id)
        covered_turns.add(turn_idx)
        selected_stage[candidate_id] = "turn_coverage"
        coverage_added += 1

    backfill_added = 0
    for cand in qk_ranked:
        if len(selected) >= target_k:
            break
        candidate_id = int(cand["candidate_id"])
        if candidate_id in selected_ids:
            continue
        selected.append(cand)
        selected_ids.add(candidate_id)
        selected_stage[candidate_id] = "qk_backfill"
        backfill_added += 1

    debug = {
        "selection_mode": "hybrid",
        "target_k": int(target_k),
        "route_hybrid_core_ratio": float(core_ratio),
        "route_hybrid_core_max_per_turn": int(core_max_per_turn),
        "qk_core_target_count": int(core_count),
        "qk_core_count": int(len([stage for stage in selected_stage.values() if stage == "qk_core"])),
        "turn_coverage_added": int(coverage_added),
        "qk_backfill_added": int(backfill_added),
        "selected_turn_count": int(len({int(c["turn_idx"]) for c in selected})),
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "stage": selected_stage.get(int(cand["candidate_id"]), ""),
                "score": float(cand.get("score", 0.0)),
            }
            for cand in selected[:20]
        ],
        "turn_rerank_preview": turn_debug.get("turn_rerank_preview", []),
    }
    return selected, debug


def select_candidates_evidence_pack(
    scored_candidates,
    per_head_scores,
    candidate_prefilter_scores,
    transcripts,
    query_text,
    effective_route_top_k,
    args,
):
    """Build an answer-facing evidence pack: anchors plus local support.

    The selected candidates are still the units charged to transfer accounting.
    This keeps answer-context organization and simulated KV transfer aligned.
    """
    if not scored_candidates:
        return [], {"selection_mode": "evidence_pack", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    query_type = classify_query_budget_type(query_text)
    if query_type == "summary":
        default_anchor_count = 4
        default_support_radius = 1
        default_max_turns = 12
    elif query_type == "detail":
        default_anchor_count = 3
        default_support_radius = 1
        default_max_turns = 8
    else:
        default_anchor_count = 3
        default_support_radius = 1
        default_max_turns = 10

    anchor_count = max(
        1,
        int(getattr(args, "route_pack_anchor_count", default_anchor_count)),
    )
    anchor_count = min(anchor_count, target_k)
    support_radius = max(
        0,
        int(getattr(args, "route_pack_support_radius", default_support_radius)),
    )
    max_turns = max(
        1,
        int(getattr(args, "route_pack_max_turns", default_max_turns)),
    )
    max_candidates = max(
        anchor_count,
        int(getattr(args, "route_pack_max_candidates", target_k)),
    )
    max_candidates = min(max_candidates, len(scored_candidates))
    allow_support_same_turn = bool(
        getattr(args, "route_pack_support_same_turn", True)
    )
    support_score_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "route_pack_support_score_ratio", 0.55))),
    )
    min_support_score = float(getattr(args, "route_pack_min_support_score", -1e9))

    qk_ranked = sorted(
        scored_candidates,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )
    anchors = qk_ranked[:anchor_count]
    anchor_turns = {int(cand["turn_idx"]) for cand in anchors}
    anchor_ids = {int(cand["candidate_id"]) for cand in anchors}
    top_anchor_score = max(float(cand.get("score", 0.0)) for cand in anchors)
    support_threshold = max(min_support_score, top_anchor_score * support_score_ratio)

    by_turn = defaultdict(list)
    for cand in scored_candidates:
        by_turn[int(cand["turn_idx"])].append(cand)
    for turn_idx in list(by_turn.keys()):
        by_turn[turn_idx] = sorted(
            by_turn[turn_idx],
            key=lambda cand: (
                -float(cand.get("score", 0.0)),
                int(cand["start_t"]),
                int(cand["candidate_id"]),
            ),
        )

    selected = []
    selected_ids = set()
    selected_stage = {}

    def add_candidate(cand, stage):
        candidate_id = int(cand["candidate_id"])
        if candidate_id in selected_ids:
            return False
        if len(selected) >= max_candidates:
            return False
        selected.append(cand)
        selected_ids.add(candidate_id)
        selected_stage[candidate_id] = stage
        return True

    for cand in anchors:
        add_candidate(cand, "anchor")

    support_added = 0
    if support_radius > 0 and len(selected) < max_candidates:
        candidate_by_turn_chunk = {
            (int(cand["turn_idx"]), int(cand["local_chunk_idx"])): cand
            for cand in scored_candidates
        }
        for anchor in anchors:
            if len(selected) >= max_candidates:
                break
            anchor_turn = int(anchor["turn_idx"])
            anchor_chunk = int(anchor["local_chunk_idx"])
            for delta in range(1, support_radius + 1):
                for key in (
                    (anchor_turn, anchor_chunk - delta),
                    (anchor_turn, anchor_chunk + delta),
                ):
                    support = candidate_by_turn_chunk.get(key)
                    if support is None:
                        continue
                    if float(support.get("score", 0.0)) < support_threshold:
                        continue
                    if add_candidate(support, "neighbor_support"):
                        support_added += 1
                    if len(selected) >= max_candidates:
                        break
                if len(selected) >= max_candidates:
                    break

    same_turn_added = 0
    if allow_support_same_turn and len(selected) < max_candidates:
        for turn_idx in sorted(anchor_turns):
            if len(selected) >= max_candidates:
                break
            for support in by_turn.get(turn_idx, []):
                if float(support.get("score", 0.0)) < support_threshold:
                    continue
                if add_candidate(support, "same_turn_support"):
                    same_turn_added += 1
                break

    coverage_added = 0
    if len(selected) < max_candidates:
        turn_candidates, turn_debug = select_candidates_turn_rerank(
            scored_candidates,
            per_head_scores,
            candidate_prefilter_scores,
            transcripts,
            query_text,
            len(scored_candidates),
            args,
        )
        covered_turns = {int(cand["turn_idx"]) for cand in selected}
        for cand in turn_candidates:
            if len(selected) >= max_candidates:
                break
            turn_idx = int(cand["turn_idx"])
            if turn_idx in covered_turns:
                continue
            if len(covered_turns) >= max_turns:
                break
            if add_candidate(cand, "coverage_turn"):
                covered_turns.add(turn_idx)
                coverage_added += 1
    else:
        turn_debug = {"turn_rerank_preview": []}

    backfill_added = 0
    for cand in qk_ranked:
        if len(selected) >= min(target_k, max_candidates):
            break
        if len({int(item["turn_idx"]) for item in selected}) >= max_turns:
            break
        if add_candidate(cand, "qk_backfill"):
            backfill_added += 1

    # Answer order: expose a few anchors first, then keep the rest chronological
    # so the generator sees both salience and conversational flow.
    anchor_order = [
        int(cand["candidate_id"])
        for cand in sorted(
            [cand for cand in selected if selected_stage.get(int(cand["candidate_id"])) == "anchor"],
            key=lambda cand: -float(cand.get("score", 0.0)),
        )
    ]
    anchor_order_set = set(anchor_order)
    tail = [
        cand for cand in selected
        if int(cand["candidate_id"]) not in anchor_order_set
    ]
    ordered_selected = [
        next(cand for cand in selected if int(cand["candidate_id"]) == candidate_id)
        for candidate_id in anchor_order
    ] + sorted(
        tail,
        key=lambda cand: (
            int(cand["start_t"]),
            int(cand["turn_idx"]),
            -float(cand.get("score", 0.0)),
        ),
    )

    debug = {
        "selection_mode": "evidence_pack",
        "target_k": int(target_k),
        "query_budget_type": query_type,
        "route_pack_anchor_count": int(anchor_count),
        "route_pack_support_radius": int(support_radius),
        "route_pack_max_turns": int(max_turns),
        "route_pack_max_candidates": int(max_candidates),
        "route_pack_support_score_ratio": float(support_score_ratio),
        "route_pack_support_threshold": float(support_threshold),
        "anchor_count": int(len(anchors)),
        "neighbor_support_added": int(support_added),
        "same_turn_support_added": int(same_turn_added),
        "coverage_turn_added": int(coverage_added),
        "qk_backfill_added": int(backfill_added),
        "selected_turn_count": int(len({int(c["turn_idx"]) for c in ordered_selected})),
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "stage": selected_stage.get(int(cand["candidate_id"]), ""),
                "score": float(cand.get("score", 0.0)),
            }
            for cand in ordered_selected[:30]
        ],
        "turn_rerank_preview": turn_debug.get("turn_rerank_preview", []),
    }
    return ordered_selected, debug


def select_candidates_evidence_pack_v2(
    scored_candidates,
    per_head_scores,
    candidate_prefilter_scores,
    transcripts,
    query_text,
    effective_route_top_k,
    args,
):
    """Build a support-first evidence pack around Q-K anchors.

    v1 often collapsed into anchor + turn coverage because support candidates
    had to pass a high relative Q-K threshold.  v2 treats nearby evidence as a
    separate budget: it first anchors on high Q-K chunks, then pulls in adjacent
    chunks/turns that are already in the scored candidate pool, and only then
    spends the remaining slots on broader turn coverage or Q-K backfill.
    """
    if not scored_candidates:
        return [], {"selection_mode": "evidence_pack_v2", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    query_type = classify_query_budget_type(query_text)
    anchor_count = max(
        1,
        int(getattr(args, "route_pack_anchor_count", 3)),
    )
    anchor_count = min(anchor_count, target_k)
    support_radius = max(
        0,
        int(getattr(args, "route_pack_support_radius", 1)),
    )
    max_candidates = max(
        anchor_count,
        int(getattr(args, "route_pack_max_candidates", target_k)),
    )
    max_candidates = min(max_candidates, len(scored_candidates))
    max_turns = max(
        1,
        int(getattr(args, "route_pack_max_turns", max_candidates)),
    )
    support_score_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "route_pack_support_score_ratio", 0.0))),
    )
    min_support_score = float(getattr(args, "route_pack_min_support_score", -1e9))
    allow_support_same_turn = bool(
        getattr(args, "route_pack_support_same_turn", True)
    )

    qk_ranked = sorted(
        scored_candidates,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )
    anchors = qk_ranked[:anchor_count]
    top_anchor_score = max(float(cand.get("score", 0.0)) for cand in anchors)
    if support_score_ratio <= 0.0:
        support_threshold = min_support_score
    else:
        support_threshold = max(min_support_score, top_anchor_score * support_score_ratio)

    if query_type == "summary":
        support_budget = max(anchor_count, int(round(max_candidates * 0.35)))
        coverage_budget = max(2, int(round(max_candidates * 0.45)))
    elif query_type == "detail":
        support_budget = max(anchor_count, int(round(max_candidates * 0.50)))
        coverage_budget = max(1, int(round(max_candidates * 0.20)))
    else:
        support_budget = max(anchor_count, int(round(max_candidates * 0.45)))
        coverage_budget = max(1, int(round(max_candidates * 0.30)))
    support_budget = min(max_candidates, support_budget)
    coverage_budget = min(max_candidates, coverage_budget)

    by_turn = defaultdict(list)
    candidate_by_turn_chunk = {}
    for cand in scored_candidates:
        turn_idx = int(cand["turn_idx"])
        by_turn[turn_idx].append(cand)
        candidate_by_turn_chunk[(turn_idx, int(cand["local_chunk_idx"]))] = cand
    for turn_idx in list(by_turn.keys()):
        by_turn[turn_idx] = sorted(
            by_turn[turn_idx],
            key=lambda cand: (
                -float(cand.get("score", 0.0)),
                int(cand["start_t"]),
                int(cand["candidate_id"]),
            ),
        )

    selected = []
    selected_ids = set()
    selected_stage = {}

    def add_candidate(cand, stage):
        candidate_id = int(cand["candidate_id"])
        if candidate_id in selected_ids:
            return False
        if len(selected) >= max_candidates:
            return False
        selected.append(cand)
        selected_ids.add(candidate_id)
        selected_stage[candidate_id] = stage
        return True

    def is_support_candidate(cand):
        return float(cand.get("score", 0.0)) >= support_threshold

    for cand in anchors:
        add_candidate(cand, "anchor")

    neighbor_support_added = 0
    same_turn_support_added = 0
    adjacent_turn_support_added = 0
    support_candidates_seen = 0
    support_candidates_below_threshold = 0

    for anchor in anchors:
        if len(selected) >= support_budget:
            break
        anchor_turn = int(anchor["turn_idx"])
        anchor_chunk = int(anchor["local_chunk_idx"])
        for delta in range(1, support_radius + 1):
            for key in (
                (anchor_turn, anchor_chunk - delta),
                (anchor_turn, anchor_chunk + delta),
            ):
                if len(selected) >= support_budget:
                    break
                support = candidate_by_turn_chunk.get(key)
                if support is None:
                    continue
                support_candidates_seen += 1
                if not is_support_candidate(support):
                    support_candidates_below_threshold += 1
                    continue
                if add_candidate(support, "neighbor_support"):
                    neighbor_support_added += 1

    if allow_support_same_turn:
        for anchor in anchors:
            if len(selected) >= support_budget:
                break
            for support in by_turn.get(int(anchor["turn_idx"]), []):
                if len(selected) >= support_budget:
                    break
                if int(support["candidate_id"]) == int(anchor["candidate_id"]):
                    continue
                support_candidates_seen += 1
                if not is_support_candidate(support):
                    support_candidates_below_threshold += 1
                    continue
                if add_candidate(support, "same_turn_support"):
                    same_turn_support_added += 1
                    break

    if support_radius > 0:
        for anchor in anchors:
            if len(selected) >= support_budget:
                break
            anchor_turn = int(anchor["turn_idx"])
            for delta in range(1, support_radius + 1):
                for turn_idx in (anchor_turn - delta, anchor_turn + delta):
                    if len(selected) >= support_budget:
                        break
                    turn_candidates = by_turn.get(int(turn_idx), [])
                    if not turn_candidates:
                        continue
                    support = turn_candidates[0]
                    support_candidates_seen += 1
                    if not is_support_candidate(support):
                        support_candidates_below_threshold += 1
                        continue
                    if add_candidate(support, "adjacent_turn_support"):
                        adjacent_turn_support_added += 1

    coverage_added = 0
    turn_debug = {"turn_rerank_preview": []}
    remaining_coverage_budget = max(0, min(coverage_budget, max_candidates - len(selected)))
    if remaining_coverage_budget > 0 and len(selected) < max_candidates:
        turn_candidates, turn_debug = select_candidates_turn_rerank(
            scored_candidates,
            per_head_scores,
            candidate_prefilter_scores,
            transcripts,
            query_text,
            len(scored_candidates),
            args,
        )
        covered_turns = {int(cand["turn_idx"]) for cand in selected}
        for cand in turn_candidates:
            if coverage_added >= remaining_coverage_budget:
                break
            if len(selected) >= max_candidates:
                break
            turn_idx = int(cand["turn_idx"])
            if turn_idx in covered_turns:
                continue
            if len(covered_turns) >= max_turns:
                break
            if add_candidate(cand, "coverage_turn"):
                covered_turns.add(turn_idx)
                coverage_added += 1

    backfill_added = 0
    for cand in qk_ranked:
        if len(selected) >= min(target_k, max_candidates):
            break
        if len({int(item["turn_idx"]) for item in selected}) >= max_turns:
            break
        if add_candidate(cand, "qk_backfill"):
            backfill_added += 1

    # Preserve salient anchors at the front, then give the remaining support in
    # chronological order so answer generation can read it as conversation.
    anchor_order = [
        int(cand["candidate_id"])
        for cand in sorted(
            [
                cand
                for cand in selected
                if selected_stage.get(int(cand["candidate_id"])) == "anchor"
            ],
            key=lambda cand: -float(cand.get("score", 0.0)),
        )
    ]
    anchor_order_set = set(anchor_order)
    tail = [
        cand for cand in selected
        if int(cand["candidate_id"]) not in anchor_order_set
    ]
    ordered_selected = [
        next(cand for cand in selected if int(cand["candidate_id"]) == candidate_id)
        for candidate_id in anchor_order
    ] + sorted(
        tail,
        key=lambda cand: (
            int(cand["start_t"]),
            int(cand["turn_idx"]),
            -float(cand.get("score", 0.0)),
        ),
    )

    debug = {
        "selection_mode": "evidence_pack_v2",
        "target_k": int(target_k),
        "query_budget_type": query_type,
        "route_pack_anchor_count": int(anchor_count),
        "route_pack_support_radius": int(support_radius),
        "route_pack_max_turns": int(max_turns),
        "route_pack_max_candidates": int(max_candidates),
        "route_pack_support_score_ratio": float(support_score_ratio),
        "route_pack_support_threshold": float(support_threshold),
        "support_budget": int(support_budget),
        "coverage_budget": int(coverage_budget),
        "anchor_count": int(len(anchors)),
        "neighbor_support_added": int(neighbor_support_added),
        "same_turn_support_added": int(same_turn_support_added),
        "adjacent_turn_support_added": int(adjacent_turn_support_added),
        "support_total_added": int(
            neighbor_support_added
            + same_turn_support_added
            + adjacent_turn_support_added
        ),
        "support_candidates_seen": int(support_candidates_seen),
        "support_candidates_below_threshold": int(support_candidates_below_threshold),
        "coverage_turn_added": int(coverage_added),
        "qk_backfill_added": int(backfill_added),
        "selected_turn_count": int(len({int(c["turn_idx"]) for c in ordered_selected})),
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "stage": selected_stage.get(int(cand["candidate_id"]), ""),
                "score": float(cand.get("score", 0.0)),
            }
            for cand in ordered_selected[:30]
        ],
        "turn_rerank_preview": turn_debug.get("turn_rerank_preview", []),
    }
    return ordered_selected, debug


def select_candidates_evidence_pack_v3(
    scored_candidates,
    per_head_scores,
    candidate_prefilter_scores,
    transcripts,
    query_text,
    effective_route_top_k,
    args,
):
    """Keep the current Q-K core, then make small support swaps.

    v2 made evidence cheaper by selecting fewer chunks, but lost turn recall.
    v3 is the conservative variant: start from the same chunk_topk set as the
    current profile, then replace only low-value duplicate-turn tail chunks with
    local support around the strongest anchors.  This asks whether local support
    helps answer quality without shrinking the selected evidence set.
    """
    if not scored_candidates:
        return [], {"selection_mode": "evidence_pack_v3", "target_k": 0}

    target_k = min(int(effective_route_top_k), len(scored_candidates))
    support_radius = max(
        0,
        int(getattr(args, "route_pack_support_radius", 1)),
    )
    anchor_count = max(1, int(getattr(args, "route_pack_anchor_count", 3)))
    anchor_count = min(anchor_count, target_k)
    max_candidates = max(
        target_k,
        int(getattr(args, "route_pack_max_candidates", target_k)),
    )
    max_candidates = min(max_candidates, len(scored_candidates))
    support_score_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "route_pack_support_score_ratio", 0.0))),
    )
    min_support_score = float(getattr(args, "route_pack_min_support_score", -1e9))
    allow_support_same_turn = bool(
        getattr(args, "route_pack_support_same_turn", True)
    )

    qk_core, qk_core_debug = select_candidates_chunk_topk(
        scored_candidates,
        per_head_scores,
        target_k,
        bool(getattr(args, "route_per_head", False)),
    )
    qk_core = sorted(
        qk_core,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )[:target_k]
    qk_ranked = sorted(
        scored_candidates,
        key=lambda cand: (
            -float(cand.get("score", 0.0)),
            int(cand["start_t"]),
            int(cand["candidate_id"]),
        ),
    )
    anchors = qk_ranked[:anchor_count]
    top_anchor_score = max(float(cand.get("score", 0.0)) for cand in anchors)
    if support_score_ratio <= 0.0:
        support_threshold = min_support_score
    else:
        support_threshold = max(min_support_score, top_anchor_score * support_score_ratio)

    by_turn = defaultdict(list)
    candidate_by_turn_chunk = {}
    for cand in scored_candidates:
        turn_idx = int(cand["turn_idx"])
        by_turn[turn_idx].append(cand)
        candidate_by_turn_chunk[(turn_idx, int(cand["local_chunk_idx"]))] = cand
    for turn_idx in list(by_turn.keys()):
        by_turn[turn_idx] = sorted(
            by_turn[turn_idx],
            key=lambda cand: (
                -float(cand.get("score", 0.0)),
                int(cand["start_t"]),
                int(cand["candidate_id"]),
            ),
        )

    selected = list(qk_core)
    selected_ids = {int(cand["candidate_id"]) for cand in selected}
    selected_stage = {int(cand["candidate_id"]): "qk_core" for cand in selected}

    core_turn_counts = defaultdict(int)
    for cand in selected:
        core_turn_counts[int(cand["turn_idx"])] += 1

    def is_support_candidate(cand):
        return float(cand.get("score", 0.0)) >= support_threshold

    support_candidates = []
    support_seen_ids = set()
    support_candidates_seen = 0
    support_candidates_below_threshold = 0

    def add_support_candidate(cand, stage, anchor):
        nonlocal support_candidates_seen, support_candidates_below_threshold
        if cand is None:
            return
        candidate_id = int(cand["candidate_id"])
        if candidate_id in selected_ids or candidate_id in support_seen_ids:
            return
        support_candidates_seen += 1
        if not is_support_candidate(cand):
            support_candidates_below_threshold += 1
            return
        support_seen_ids.add(candidate_id)
        support_candidates.append(
            {
                "candidate": cand,
                "stage": stage,
                "anchor_id": int(anchor["candidate_id"]),
                "anchor_turn": int(anchor["turn_idx"]),
                "anchor_score": float(anchor.get("score", 0.0)),
                "score": float(cand.get("score", 0.0)),
            }
        )

    for anchor in anchors:
        anchor_turn = int(anchor["turn_idx"])
        anchor_chunk = int(anchor["local_chunk_idx"])
        for delta in range(1, support_radius + 1):
            add_support_candidate(
                candidate_by_turn_chunk.get((anchor_turn, anchor_chunk - delta)),
                "neighbor_support",
                anchor,
            )
            add_support_candidate(
                candidate_by_turn_chunk.get((anchor_turn, anchor_chunk + delta)),
                "neighbor_support",
                anchor,
            )
        if allow_support_same_turn:
            for support in by_turn.get(anchor_turn, []):
                if int(support["candidate_id"]) == int(anchor["candidate_id"]):
                    continue
                add_support_candidate(support, "same_turn_support", anchor)
                break
        for delta in range(1, support_radius + 1):
            for turn_idx in (anchor_turn - delta, anchor_turn + delta):
                turn_candidates = by_turn.get(int(turn_idx), [])
                if turn_candidates:
                    add_support_candidate(
                        turn_candidates[0],
                        "adjacent_turn_support",
                        anchor,
                    )

    support_candidates = sorted(
        support_candidates,
        key=lambda item: (
            -float(item["anchor_score"]),
            -float(item["score"]),
            int(item["candidate"]["start_t"]),
            int(item["candidate"]["candidate_id"]),
        ),
    )

    if target_k <= 8:
        replacement_budget = 1
    elif target_k <= 12:
        replacement_budget = 2
    else:
        replacement_budget = 3
    replacement_budget = min(replacement_budget, max(0, target_k - anchor_count))

    replacement_added = 0
    replacement_skipped_no_drop = 0
    dropped_candidates = []

    def find_drop_index_for_support(support):
        support_turn = int(support["turn_idx"])
        anchor_ids = {int(cand["candidate_id"]) for cand in anchors}
        eligible = []
        for idx, cand in enumerate(selected):
            candidate_id = int(cand["candidate_id"])
            if candidate_id in anchor_ids:
                continue
            turn_idx = int(cand["turn_idx"])
            duplicate_turn = core_turn_counts.get(turn_idx, 0) > 1
            same_support_turn = turn_idx == support_turn
            if not duplicate_turn and not same_support_turn:
                continue
            eligible.append(
                (
                    float(cand.get("score", 0.0)),
                    0 if duplicate_turn else 1,
                    -idx,
                    idx,
                )
            )
        if not eligible:
            return None
        return sorted(eligible)[0][3]

    for item in support_candidates:
        if replacement_added >= replacement_budget:
            break
        support = item["candidate"]
        drop_idx = find_drop_index_for_support(support)
        if drop_idx is None:
            replacement_skipped_no_drop += 1
            continue
        dropped = selected[drop_idx]
        dropped_candidates.append(
            {
                "candidate_id": int(dropped["candidate_id"]),
                "turn_idx": int(dropped["turn_idx"]),
                "score": float(dropped.get("score", 0.0)),
                "replacement_id": int(support["candidate_id"]),
                "replacement_turn_idx": int(support["turn_idx"]),
                "replacement_stage": item["stage"],
                "replacement_score": float(support.get("score", 0.0)),
            }
        )
        selected_ids.remove(int(dropped["candidate_id"]))
        core_turn_counts[int(dropped["turn_idx"])] -= 1
        selected[drop_idx] = support
        selected_ids.add(int(support["candidate_id"]))
        core_turn_counts[int(support["turn_idx"])] += 1
        selected_stage[int(support["candidate_id"])] = item["stage"]
        replacement_added += 1

    if len(selected) < min(target_k, max_candidates):
        for cand in qk_ranked:
            if len(selected) >= min(target_k, max_candidates):
                break
            candidate_id = int(cand["candidate_id"])
            if candidate_id in selected_ids:
                continue
            selected.append(cand)
            selected_ids.add(candidate_id)
            selected_stage[candidate_id] = "qk_backfill"

    anchor_order = [
        int(cand["candidate_id"])
        for cand in sorted(anchors, key=lambda cand: -float(cand.get("score", 0.0)))
        if int(cand["candidate_id"]) in selected_ids
    ]
    anchor_order_set = set(anchor_order)
    tail = [
        cand for cand in selected
        if int(cand["candidate_id"]) not in anchor_order_set
    ]
    ordered_selected = [
        next(cand for cand in selected if int(cand["candidate_id"]) == candidate_id)
        for candidate_id in anchor_order
    ] + sorted(
        tail,
        key=lambda cand: (
            int(cand["start_t"]),
            int(cand["turn_idx"]),
            -float(cand.get("score", 0.0)),
        ),
    )

    neighbor_support_added = sum(
        1
        for cand in ordered_selected
        if selected_stage.get(int(cand["candidate_id"])) == "neighbor_support"
    )
    same_turn_support_added = sum(
        1
        for cand in ordered_selected
        if selected_stage.get(int(cand["candidate_id"])) == "same_turn_support"
    )
    adjacent_turn_support_added = sum(
        1
        for cand in ordered_selected
        if selected_stage.get(int(cand["candidate_id"])) == "adjacent_turn_support"
    )
    qk_backfill_added = sum(
        1
        for cand in ordered_selected
        if selected_stage.get(int(cand["candidate_id"])) == "qk_backfill"
    )

    debug = {
        "selection_mode": "evidence_pack_v3",
        "target_k": int(target_k),
        "query_budget_type": classify_query_budget_type(query_text),
        "route_pack_anchor_count": int(anchor_count),
        "route_pack_support_radius": int(support_radius),
        "route_pack_max_candidates": int(max_candidates),
        "route_pack_support_score_ratio": float(support_score_ratio),
        "route_pack_support_threshold": float(support_threshold),
        "support_budget": int(replacement_budget),
        "coverage_budget": 0,
        "anchor_count": int(len(anchors)),
        "qk_core_count": int(len(qk_core)),
        "replacement_budget": int(replacement_budget),
        "replacement_added": int(replacement_added),
        "replacement_skipped_no_drop": int(replacement_skipped_no_drop),
        "neighbor_support_added": int(neighbor_support_added),
        "same_turn_support_added": int(same_turn_support_added),
        "adjacent_turn_support_added": int(adjacent_turn_support_added),
        "support_total_added": int(
            neighbor_support_added
            + same_turn_support_added
            + adjacent_turn_support_added
        ),
        "support_candidates_seen": int(support_candidates_seen),
        "support_candidates_below_threshold": int(support_candidates_below_threshold),
        "coverage_turn_added": 0,
        "qk_backfill_added": int(qk_backfill_added),
        "selected_turn_count": int(len({int(c["turn_idx"]) for c in ordered_selected})),
        "dropped_candidates": dropped_candidates[:20],
        "qk_core_debug": qk_core_debug,
        "selected_candidate_stages": [
            {
                "candidate_id": int(cand["candidate_id"]),
                "turn_idx": int(cand["turn_idx"]),
                "stage": selected_stage.get(int(cand["candidate_id"]), ""),
                "score": float(cand.get("score", 0.0)),
            }
            for cand in ordered_selected[:30]
        ],
    }
    return ordered_selected, debug


def _answer_cue_bonus(text, query_type):
    text = (text or "").lower()
    detail_cues = [
        "decision",
        "decide",
        "decided",
        "agree",
        "agreed",
        "think",
        "thought",
        "suggest",
        "suggested",
        "recommend",
        "prefer",
        "should",
        "could",
        "number",
        "because",
        "yes",
        "no",
    ]
    summary_cues = [
        "discuss",
        "discussed",
        "summary",
        "overview",
        "presentation",
        "main",
        "plan",
        "issue",
        "proposal",
    ]
    balanced_cues = [
        "decide",
        "agree",
        "think",
        "summary",
        "main",
    ]

    if query_type == "detail":
        cues = detail_cues
    elif query_type == "summary":
        cues = summary_cues
    else:
        cues = balanced_cues

    return float(sum(1 for cue in cues if cue in text))


def order_candidates_for_answer_aware(candidates, transcripts, query_text):
    """Re-rank selected evidence with a light answer-oriented turn scorer."""
    if not candidates:
        return [], []

    query_type = classify_query_budget_type(query_text)
    query_tokens = set(tokenize_lexical_text(query_text))
    query_text_lc = (query_text or "").lower()

    by_turn = defaultdict(list)
    for cand in candidates:
        by_turn[int(cand["turn_idx"])].append(cand)

    turn_debug = []
    raw_scores = []
    for turn_idx, turn_cands in by_turn.items():
        if 0 <= turn_idx < len(transcripts):
            turn = transcripts[turn_idx]
            speaker = turn.get("speaker", "").strip() or "Speaker"
            content = turn.get("content", "").strip()
            turn_text = f"{speaker}: {content}"
        else:
            speaker = "Speaker"
            turn_text = ""

        turn_tokens = set(tokenize_lexical_text(turn_text))
        lexical_overlap = 0.0
        if query_tokens:
            lexical_overlap = float(len(query_tokens & turn_tokens)) / float(len(query_tokens))

        speaker_bonus = 1.0 if speaker.lower() in query_text_lc else 0.0
        cue_bonus = _answer_cue_bonus(turn_text, query_type)
        max_qk = max(float(c.get("score", 0.0)) for c in turn_cands)
        mean_qk = float(np.mean([float(c.get("score", 0.0)) for c in turn_cands]))

        raw_scores.append(max_qk)
        turn_debug.append(
            {
                "turn_idx": int(turn_idx),
                "speaker": speaker,
                "start_t": min(int(c["start_t"]) for c in turn_cands),
                "max_qk": float(max_qk),
                "mean_qk": float(mean_qk),
                "lexical_overlap": float(lexical_overlap),
                "speaker_bonus": float(speaker_bonus),
                "cue_bonus": float(cue_bonus),
                "num_chunks": len(turn_cands),
            }
        )

    min_qk = min(raw_scores) if raw_scores else 0.0
    max_qk = max(raw_scores) if raw_scores else 0.0
    qk_span = max(max_qk - min_qk, 1e-6)
    for item in turn_debug:
        norm_qk = (float(item["max_qk"]) - min_qk) / qk_span
        item["answer_score"] = float(
            0.55 * norm_qk
            + 0.25 * float(item["lexical_overlap"])
            + 0.12 * min(1.0, float(item["cue_bonus"]) / 2.0)
            + 0.08 * float(item["speaker_bonus"])
        )

    ranked_turns = sorted(
        turn_debug,
        key=lambda x: (-float(x["answer_score"]), int(x["start_t"]), int(x["turn_idx"])),
    )

    top_count = min(4, len(ranked_turns))
    top_turn_ids = [int(item["turn_idx"]) for item in ranked_turns[:top_count]]
    top_turn_set = set(top_turn_ids)
    tail_turns = sorted(
        [item for item in turn_debug if int(item["turn_idx"]) not in top_turn_set],
        key=lambda x: (int(x["start_t"]), int(x["turn_idx"])),
    )
    ordered_turn_ids = top_turn_ids + [int(item["turn_idx"]) for item in tail_turns]
    turn_rank = {int(turn_idx): rank for rank, turn_idx in enumerate(ordered_turn_ids)}

    ordered_candidates = sorted(
        candidates,
        key=lambda c: (
            turn_rank.get(int(c["turn_idx"]), 10**9),
            -float(c.get("score", 0.0)),
            int(c["start_t"]),
        ),
    )

    ranked_turn_preview = [
        {
            "turn_idx": int(item["turn_idx"]),
            "speaker": item["speaker"],
            "answer_score": float(item["answer_score"]),
            "max_qk": float(item["max_qk"]),
            "lexical_overlap": float(item["lexical_overlap"]),
            "cue_bonus": float(item["cue_bonus"]),
            "speaker_bonus": float(item["speaker_bonus"]),
            "num_chunks": int(item["num_chunks"]),
        }
        for item in ranked_turns[:12]
    ]
    return ordered_candidates, ranked_turn_preview


def build_hierarchical_candidates(turn_boundaries, total_tokens, turn_to_topic_ids, chunk_size):
    candidates = []
    for turn_idx, (start_t, end_t) in enumerate(turn_boundaries):
        if end_t <= start_t or start_t >= total_tokens:
            continue

        topic_ids = turn_to_topic_ids[turn_idx]
        if not topic_ids:
            continue

        local_chunk_idx = 0
        for c_start in range(start_t, min(end_t, total_tokens), chunk_size):
            c_end = min(c_start + chunk_size, end_t, total_tokens)
            if c_end <= c_start:
                continue
            candidates.append(
                {
                    "candidate_id": len(candidates),
                    "turn_idx": int(turn_idx),
                    "local_chunk_idx": int(local_chunk_idx),
                    "topic_ids": [int(x) for x in topic_ids],
                    "start_t": int(c_start),
                    "end_t": int(c_end),
                    "n_tokens": int(c_end - c_start),
                }
            )
            local_chunk_idx += 1

    return candidates


def sample_topic_turn_indices(turns, max_turns):
    if not turns:
        return []
    if len(turns) <= max_turns:
        return [int(t) for t in turns]
    if max_turns <= 1:
        return [int(turns[len(turns) // 2])]

    positions = np.linspace(0, len(turns) - 1, num=max_turns)
    sampled = []
    used = set()
    for pos in positions:
        idx = int(round(float(pos)))
        idx = max(0, min(idx, len(turns) - 1))
        turn_idx = int(turns[idx])
        if turn_idx not in used:
            sampled.append(turn_idx)
            used.add(turn_idx)
    return sampled


def _load_manual_topic_node_layout(layout_path):
    if not layout_path:
        return None
    path = os.path.expanduser(str(layout_path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"topic-node layout file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "topic_to_node" in payload:
        payload = payload["topic_to_node"]

    mapping = {}
    if isinstance(payload, dict):
        for topic_id, node_id in payload.items():
            mapping[int(topic_id)] = int(node_id)
    elif isinstance(payload, list):
        for item in payload:
            mapping[int(item["topic_id"])] = int(item["node_id"])
    else:
        raise ValueError(
            "topic-node layout must be a dict, a list of {topic_id,node_id}, "
            "or {'topic_to_node': {...}}"
        )
    return mapping


def assign_topics_to_virtual_nodes(
    topic_nodes,
    num_nodes,
    assignment_mode="contiguous",
    layout_path=None,
):
    """Assign semantic topics to virtual nodes for distributed-fetch simulation."""
    topic_ids = [int(topic["topic_id"]) for topic in topic_nodes]
    if not topic_ids:
        return {}, {}

    configured_num_nodes = max(1, int(num_nodes))
    effective_num_nodes = min(configured_num_nodes, max(1, len(topic_ids)))
    assignment_mode = (assignment_mode or "contiguous").lower()
    if assignment_mode not in VALID_NODE_ASSIGNMENT_MODES:
        raise ValueError(
            "unknown node_assignment_mode="
            f"{assignment_mode!r}; expected one of {sorted(VALID_NODE_ASSIGNMENT_MODES)}"
        )

    topic_to_node_id = {}
    node_to_topic_ids = defaultdict(list)

    if assignment_mode == "manual":
        manual_mapping = _load_manual_topic_node_layout(layout_path)
        if not manual_mapping:
            raise ValueError(
                "node_assignment_mode=manual requires --topic_node_layout_path"
            )
        for topic_id in topic_ids:
            if topic_id not in manual_mapping:
                raise ValueError(
                    f"manual topic-node layout missing topic_id={topic_id}"
                )
            node_id = int(manual_mapping[topic_id])
            if node_id < 0 or node_id >= configured_num_nodes:
                raise ValueError(
                    f"manual topic-node layout maps topic_id={topic_id} "
                    f"to node_id={node_id}, outside valid range "
                    f"[0, {configured_num_nodes - 1}]"
                )
            topic_to_node_id[topic_id] = node_id
            node_to_topic_ids[node_id].append(topic_id)
    elif assignment_mode == "round_robin":
        for idx, topic_id in enumerate(topic_ids):
            node_id = int(idx % effective_num_nodes)
            topic_to_node_id[topic_id] = node_id
            node_to_topic_ids[node_id].append(topic_id)
    else:
        # Contiguous assignment keeps nearby semantic topics on the same node.
        boundaries = np.linspace(0, len(topic_ids), num=effective_num_nodes + 1)
        for node_id in range(effective_num_nodes):
            start = int(round(float(boundaries[node_id])))
            end = int(round(float(boundaries[node_id + 1])))
            if node_id == effective_num_nodes - 1:
                end = len(topic_ids)
            start = max(0, min(start, len(topic_ids)))
            end = max(start + 1, min(end, len(topic_ids)))
            for topic_id in topic_ids[start:end]:
                topic_to_node_id[int(topic_id)] = int(node_id)
                node_to_topic_ids[int(node_id)].append(int(topic_id))

    for topic_id in topic_ids:
        if topic_id not in topic_to_node_id:
            node_id = int(len(topic_to_node_id) % effective_num_nodes)
            topic_to_node_id[topic_id] = node_id
            node_to_topic_ids[node_id].append(topic_id)

    node_to_topic_ids = {
        int(node_id): sorted(int(topic_id) for topic_id in topic_list)
        for node_id, topic_list in sorted(node_to_topic_ids.items())
    }
    return topic_to_node_id, node_to_topic_ids


def build_topic_speaker_summary(topic, transcripts, max_speakers=3):
    speaker_counts = defaultdict(int)
    for turn_idx in topic.get("turns", []):
        if 0 <= int(turn_idx) < len(transcripts):
            speaker = transcripts[int(turn_idx)].get("speaker", "").strip() or "Speaker"
            speaker_counts[speaker] += 1

    ranked = sorted(speaker_counts.items(), key=lambda x: (-x[1], x[0]))
    return ranked[: max(1, int(max_speakers))]


def build_mainline_topic_index_qmsum(
    meeting,
    transcripts,
    topic_prototype_turns=3,
    topic_representation_template="basic",
):
    topic_nodes, turn_to_topic_ids = build_topic_nodes(meeting, num_turns=len(transcripts))
    topic_turn_info = {topic["topic_id"]: topic["turns"] for topic in topic_nodes}

    topic_repr_turns = {}
    topic_repr_entries = {}
    for topic in topic_nodes:
        topic_id = int(topic["topic_id"])
        sampled_turns = sample_topic_turn_indices(
            topic.get("turns", []),
            max(1, int(topic_prototype_turns)),
        )
        topic_repr_turns[topic_id] = sampled_turns

        entry_list = []
        label = topic.get("label", "").strip()
        if label:
            entry_list.append(
                {
                    "entry_type": "label",
                    "text": label,
                    "turn_idx": None,
                }
            )

        for turn_idx in sampled_turns:
            if 0 <= int(turn_idx) < len(transcripts):
                turn = transcripts[int(turn_idx)]
                speaker = turn.get("speaker", "").strip() or "Speaker"
                content = turn.get("content", "").strip()
                entry_list.append(
                    {
                        "entry_type": "turn",
                        "text": f"{speaker}: {content}",
                        "turn_idx": int(turn_idx),
                    }
                )

        if topic_representation_template == "enhanced":
            top_speakers = build_topic_speaker_summary(topic, transcripts, max_speakers=3)
            if top_speakers:
                speaker_text = ", ".join(
                    f"{speaker} ({count})" for speaker, count in top_speakers
                )
                entry_list.append(
                    {
                        "entry_type": "speakers",
                        "text": f"Main speakers: {speaker_text}",
                        "turn_idx": None,
                    }
                )

        topic_repr_entries[topic_id] = entry_list

    return {
        "source": "precomputed_topic_text",
        "topic_nodes": topic_nodes,
        "turn_to_topic_ids": turn_to_topic_ids,
        "topic_turn_info": topic_turn_info,
        "topic_repr_turns": topic_repr_turns,
        "topic_repr_entries": topic_repr_entries,
    }


def build_topic_lexical_document(topic, transcripts, topic_index=None, label_repeat=3):
    topic_id = int(topic["topic_id"])
    tokens = []

    if topic_index is not None:
        entry_list = topic_index.get("topic_repr_entries", {}).get(topic_id, [])
        for entry in entry_list:
            entry_tokens = tokenize_lexical_text(entry.get("text", ""))
            if not entry_tokens:
                continue
            if entry.get("entry_type") == "label":
                tokens.extend(entry_tokens * max(1, int(label_repeat)))
            else:
                tokens.extend(entry_tokens)
        if tokens:
            return tokens

    label = topic.get("label", "").strip()
    if label:
        tokens.extend(tokenize_lexical_text(label) * max(1, int(label_repeat)))

    for turn_idx in topic.get("turns", []):
        if 0 <= int(turn_idx) < len(transcripts):
            turn = transcripts[int(turn_idx)]
            speaker = turn.get("speaker", "").strip() or "Speaker"
            content = turn.get("content", "").strip()
            tokens.extend(tokenize_lexical_text(f"{speaker}: {content}"))

    return tokens


def build_topic_lexical_stats(topic_nodes, transcripts, topic_index=None, label_repeat=3):
    doc_tf = {}
    doc_freq = defaultdict(int)
    doc_lens = {}

    for topic in topic_nodes:
        topic_id = int(topic["topic_id"])
        tokens = build_topic_lexical_document(
            topic,
            transcripts,
            topic_index=topic_index,
            label_repeat=label_repeat,
        )
        tf = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        doc_tf[topic_id] = dict(tf)
        doc_lens[topic_id] = len(tokens)
        for tok in tf.keys():
            doc_freq[tok] += 1

    avg_doc_len = float(np.mean(list(doc_lens.values()))) if doc_lens else 0.0
    num_docs = max(1, len(topic_nodes))
    idf = {}
    for tok, freq in doc_freq.items():
        idf[tok] = float(np.log(1.0 + ((num_docs - freq + 0.5) / (freq + 0.5))))

    return {
        "doc_tf": doc_tf,
        "doc_lens": doc_lens,
        "avg_doc_len": avg_doc_len,
        "idf": idf,
    }


def build_candidate_lexical_document(candidate, transcripts, topic_nodes_by_id=None):
    turn_idx = int(candidate.get("turn_idx", -1))
    tokens = []

    if 0 <= turn_idx < len(transcripts):
        turn = transcripts[turn_idx]
        speaker = turn.get("speaker", "").strip() or "Speaker"
        content = turn.get("content", "").strip()
        tokens.extend(tokenize_lexical_text(f"{speaker}: {content}"))

    if topic_nodes_by_id is not None:
        for topic_id in candidate.get("topic_ids", []):
            topic = topic_nodes_by_id.get(int(topic_id))
            if not topic:
                continue
            label = topic.get("label", "").strip()
            if label:
                tokens.extend(tokenize_lexical_text(label))

    return tokens


def build_candidate_lexical_stats(route_candidates, transcripts, topic_nodes):
    candidate_tf = {}
    candidate_lens = {}
    doc_freq = defaultdict(int)
    topic_nodes_by_id = {
        int(topic["topic_id"]): topic
        for topic in topic_nodes
    }

    for cand in route_candidates:
        candidate_id = int(cand["candidate_id"])
        tokens = build_candidate_lexical_document(
            cand,
            transcripts,
            topic_nodes_by_id=topic_nodes_by_id,
        )
        tf = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        candidate_tf[candidate_id] = dict(tf)
        candidate_lens[candidate_id] = len(tokens)
        for tok in tf.keys():
            doc_freq[tok] += 1

    avg_doc_len = (
        float(np.mean(list(candidate_lens.values())))
        if candidate_lens
        else 0.0
    )
    num_docs = max(1, len(route_candidates))
    idf = {}
    for tok, freq in doc_freq.items():
        idf[tok] = float(np.log(1.0 + ((num_docs - freq + 0.5) / (freq + 0.5))))

    return {
        "candidate_tf": candidate_tf,
        "candidate_lens": candidate_lens,
        "avg_doc_len": avg_doc_len,
        "idf": idf,
    }


def resolve_candidate_prefilter_size(args, effective_route_top_k, num_candidates):
    if num_candidates <= 0:
        return 0

    factor = max(1, int(getattr(args, "route_candidate_prefilter_factor", 4)))
    min_keep = max(1, int(getattr(args, "route_candidate_prefilter_min_keep", 24)))
    max_keep = int(getattr(args, "route_candidate_prefilter_max_keep", 0))
    keep_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "route_candidate_prefilter_keep_ratio", 0.0))),
    )

    keep_count = max(
        min_keep,
        int(effective_route_top_k) * factor,
        int(np.ceil(float(num_candidates) * keep_ratio)),
    )
    if max_keep > 0:
        keep_count = min(keep_count, max_keep)
    return min(max(1, keep_count), int(num_candidates))


def resolve_candidate_prefilter_fixed_floor(args, effective_route_top_k, num_candidates):
    if num_candidates <= 0:
        return 0

    factor = max(1, int(getattr(args, "route_candidate_prefilter_factor", 4)))
    min_keep = max(1, int(getattr(args, "route_candidate_prefilter_min_keep", 24)))
    return min(
        int(num_candidates),
        max(1, min_keep, int(effective_route_top_k) * factor),
    )


def _parse_budget_map(raw_value):
    budget_map = {}
    text = str(raw_value or "").strip()
    if not text:
        return budget_map

    for item in text.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip().lower()
        try:
            budget = int(value.strip())
        except ValueError:
            continue
        if key and budget > 0:
            budget_map[key] = budget
    return budget_map


def resolve_candidate_pool_budget(
    args,
    query_budget_type,
    effective_route_top_k,
    requested_pool_size,
):
    """Optionally shrink the exact-QK candidate pool by query type."""
    requested_pool_size = max(0, int(requested_pool_size))
    if requested_pool_size <= 0:
        return 0, "disabled", 0.0

    if not bool(getattr(args, "dynamic_candidate_pool_budget", False)):
        return requested_pool_size, "disabled", 0.0

    budget_map = _parse_budget_map(
        getattr(args, "dynamic_candidate_pool_budget_map", "")
    )
    query_type = str(query_budget_type or "balanced").lower()
    target = budget_map.get(query_type, budget_map.get("default"))
    if target is None:
        return requested_pool_size, "enabled_no_match", 0.0

    min_keep = max(
        1,
        int(getattr(args, "dynamic_candidate_pool_min_keep", 1)),
        int(effective_route_top_k),
    )
    target = max(min_keep, int(target))
    target = min(requested_pool_size, target)
    prune_ratio = 1.0 - (float(target) / max(1.0, float(requested_pool_size)))
    return int(target), f"query_type:{query_type}", float(prune_ratio)


def resolve_candidate_prefilter_min_prune_ratio(args):
    ratio = float(getattr(args, "route_candidate_prefilter_min_prune_ratio", 0.0))
    return min(1.0, max(0.0, ratio))


def score_candidates_lexical_prefilter(
    query_text,
    candidate_ids,
    candidate_lexical_stats,
):
    query_tokens = tokenize_lexical_text(query_text)
    if not query_tokens:
        return {int(candidate_id): -1e9 for candidate_id in candidate_ids}

    if not candidate_lexical_stats:
        return {int(candidate_id): -1e9 for candidate_id in candidate_ids}

    scores = {}
    for candidate_id in candidate_ids:
        candidate_id = int(candidate_id)
        score = bm25_score_tokens(
            query_tokens,
            candidate_lexical_stats["candidate_tf"].get(candidate_id, {}),
            candidate_lexical_stats["candidate_lens"].get(candidate_id, 0),
            candidate_lexical_stats["avg_doc_len"],
            candidate_lexical_stats["idf"],
        )
        scores[candidate_id] = float(score)
    return scores


def compute_adaptive_keep_ratio_from_scores(score_values, args):
    """Estimate how much evidence should survive from score uncertainty.

    A flat cheap-score distribution means the cheap scorer is uncertain, so the
    downstream exact Q-K stage should see more candidates. A peaked distribution
    means it is safer to prune harder before exact Q-K.
    """
    num_items = len(score_values)
    if num_items <= 0:
        return {
            "adaptive_keep_ratio": 0.0,
            "adaptive_uncertainty": 1.0,
            "adaptive_signal_count": 0,
            "adaptive_score_span": 0.0,
            "adaptive_reason": "empty",
            "adaptive_skip_prune": True,
        }

    min_keep_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "route_adaptive_min_keep_ratio", 0.45))),
    )
    max_keep_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "route_adaptive_max_keep_ratio", 0.90))),
    )
    if max_keep_ratio < min_keep_ratio:
        min_keep_ratio, max_keep_ratio = max_keep_ratio, min_keep_ratio

    temperature = max(
        1e-4,
        float(getattr(args, "route_adaptive_entropy_temperature", 0.25)),
    )
    min_signal_count = max(
        1,
        int(getattr(args, "route_adaptive_min_signal_count", 4)),
    )

    cleaned_scores = []
    positive_scores = []
    for value in score_values:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = -1e9
        if not np.isfinite(score) or score <= -1e8:
            score = 0.0
        else:
            score = max(0.0, score)
        cleaned_scores.append(score)
        if score > 0.0:
            positive_scores.append(score)

    signal_count = len(positive_scores)
    if signal_count < min_signal_count:
        return {
            "adaptive_keep_ratio": 1.0,
            "adaptive_uncertainty": 1.0,
            "adaptive_signal_count": int(signal_count),
            "adaptive_score_span": 0.0,
            "adaptive_reason": "no_signal",
            "adaptive_skip_prune": True,
        }

    score_span = float(max(positive_scores))
    if score_span <= 1e-8:
        return {
            "adaptive_keep_ratio": 1.0,
            "adaptive_uncertainty": 1.0,
            "adaptive_signal_count": int(signal_count),
            "adaptive_score_span": float(score_span),
            "adaptive_reason": "flat_scores",
            "adaptive_skip_prune": True,
        }

    normalized = np.asarray(cleaned_scores, dtype=np.float64) / score_span
    logits = normalized / temperature
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    denom = float(np.sum(weights))
    if denom <= 0.0 or not np.isfinite(denom):
        uncertainty = 1.0
    else:
        probs = weights / denom
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        uncertainty = entropy / max(1e-12, float(np.log(max(2, num_items))))
        uncertainty = min(1.0, max(0.0, uncertainty))

    keep_ratio = min_keep_ratio + (max_keep_ratio - min_keep_ratio) * uncertainty
    return {
        "adaptive_keep_ratio": float(min(1.0, max(0.0, keep_ratio))),
        "adaptive_uncertainty": float(uncertainty),
        "adaptive_signal_count": int(signal_count),
        "adaptive_score_span": float(score_span),
        "adaptive_reason": "entropy",
        "adaptive_skip_prune": False,
    }


def resolve_adaptive_keep_count(num_items, min_keep_count, max_keep_count, adaptive_info):
    if num_items <= 0:
        return 0
    keep_ratio = float(adaptive_info.get("adaptive_keep_ratio", 1.0))
    keep_count = int(np.ceil(float(num_items) * keep_ratio))
    keep_count = max(1, int(min_keep_count), keep_count)
    if int(max_keep_count) > 0:
        keep_count = min(keep_count, int(max_keep_count))
    return min(int(num_items), int(keep_count))


def group_candidates_into_coarse_segments(candidates, segment_size):
    if not candidates:
        return []

    segment_size = max(1, int(segment_size))
    sorted_candidates = sorted(
        candidates,
        key=lambda cand: (
            min(int(topic_id) for topic_id in cand.get("topic_ids", [-1])),
            int(cand.get("turn_idx", -1)),
            int(cand.get("local_chunk_idx", -1)),
            int(cand.get("candidate_id", -1)),
        ),
    )

    segments = []
    current = []
    current_topic_key = None
    for cand in sorted_candidates:
        topic_ids = [int(topic_id) for topic_id in cand.get("topic_ids", [])]
        topic_key = min(topic_ids) if topic_ids else -1
        if current and (topic_key != current_topic_key or len(current) >= segment_size):
            segments.append(current)
            current = []
        current.append(cand)
        current_topic_key = topic_key
    if current:
        segments.append(current)
    return segments


def apply_coarse_segment_gate(
    candidates,
    candidate_prefilter_scores,
    args,
):
    mode = str(getattr(args, "route_coarse_segment_gate", "none")).lower()
    adaptive_gate_enabled = bool(
        getattr(args, "route_adaptive_coarse_segment_gate", False)
    )
    if mode == "none" or not candidates:
        return list(candidates), {
            "route_coarse_segment_gate": mode,
            "coarse_segment_gate_before": len(candidates),
            "coarse_segment_gate_after": len(candidates),
            "coarse_segment_gate_num_segments": 0,
            "coarse_segment_gate_keep_segments": 0,
            "coarse_segment_gate_segment_size": int(
                getattr(args, "route_coarse_segment_size", 4)
            ),
            "coarse_segment_gate_min_keep": int(
                getattr(args, "route_coarse_segment_min_keep", 0)
            ),
            "coarse_segment_gate_ratio": float(
                getattr(args, "route_coarse_segment_keep_ratio", 1.0)
            ),
            "coarse_segment_gate_prune_ratio": 0.0,
            "coarse_segment_gate_ms": 0.0,
            "coarse_segment_gate_preview": [],
            "coarse_segment_gate_adaptive_enabled": bool(adaptive_gate_enabled),
            "coarse_segment_gate_adaptive_keep_ratio": 0.0,
            "coarse_segment_gate_adaptive_uncertainty": 0.0,
            "coarse_segment_gate_adaptive_reason": "disabled_or_empty",
        }

    start_time = time.perf_counter()
    segment_size = max(1, int(getattr(args, "route_coarse_segment_size", 4)))
    keep_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "route_coarse_segment_keep_ratio", 0.5))),
    )
    min_keep = max(1, int(getattr(args, "route_coarse_segment_min_keep", 32)))
    max_keep = int(getattr(args, "route_coarse_segment_max_keep", 0))
    adaptive_gate_info = {}

    segments = group_candidates_into_coarse_segments(candidates, segment_size)
    if not segments:
        return list(candidates), {
            "route_coarse_segment_gate": mode,
            "coarse_segment_gate_before": len(candidates),
            "coarse_segment_gate_after": len(candidates),
            "coarse_segment_gate_num_segments": 0,
            "coarse_segment_gate_keep_segments": 0,
            "coarse_segment_gate_segment_size": segment_size,
            "coarse_segment_gate_min_keep": min_keep,
            "coarse_segment_gate_ratio": keep_ratio,
            "coarse_segment_gate_prune_ratio": 0.0,
            "coarse_segment_gate_ms": 1000.0 * (time.perf_counter() - start_time),
            "coarse_segment_gate_preview": [],
            "coarse_segment_gate_adaptive_enabled": bool(adaptive_gate_enabled),
            "coarse_segment_gate_adaptive_keep_ratio": 0.0,
            "coarse_segment_gate_adaptive_uncertainty": 0.0,
            "coarse_segment_gate_adaptive_reason": "no_segments",
        }

    target_keep_candidates = max(min_keep, int(np.ceil(len(candidates) * keep_ratio)))
    if max_keep > 0:
        target_keep_candidates = min(target_keep_candidates, max_keep)
    target_keep_candidates = min(len(candidates), max(1, target_keep_candidates))

    if target_keep_candidates >= len(candidates) and not adaptive_gate_enabled:
        return list(candidates), {
            "route_coarse_segment_gate": f"{mode}_skipped_no_prune",
            "coarse_segment_gate_before": len(candidates),
            "coarse_segment_gate_after": len(candidates),
            "coarse_segment_gate_num_segments": len(segments),
            "coarse_segment_gate_keep_segments": len(segments),
            "coarse_segment_gate_segment_size": segment_size,
            "coarse_segment_gate_min_keep": min_keep,
            "coarse_segment_gate_ratio": keep_ratio,
            "coarse_segment_gate_prune_ratio": 0.0,
            "coarse_segment_gate_ms": 1000.0 * (time.perf_counter() - start_time),
            "coarse_segment_gate_preview": [],
            "coarse_segment_gate_adaptive_enabled": bool(adaptive_gate_enabled),
            "coarse_segment_gate_adaptive_keep_ratio": 0.0,
            "coarse_segment_gate_adaptive_uncertainty": 0.0,
            "coarse_segment_gate_adaptive_reason": "fixed_no_prune",
        }

    segment_records = []
    for segment_idx, segment in enumerate(segments):
        segment_scores = [
            float(candidate_prefilter_scores.get(int(cand["candidate_id"]), -1e9))
            for cand in segment
        ]
        finite_scores = [score for score in segment_scores if score > -1e8]
        if finite_scores:
            segment_score = float(max(finite_scores))
        else:
            # Fall back to temporal order when lexical prefilter is unavailable.
            segment_score = -float(segment_idx)
        segment_records.append(
            {
                "segment_idx": int(segment_idx),
                "score": float(segment_score),
                "start_t": min(int(cand["start_t"]) for cand in segment),
                "end_t": max(int(cand["end_t"]) for cand in segment),
                "candidate_ids": [int(cand["candidate_id"]) for cand in segment],
                "num_candidates": len(segment),
            }
        )

    if adaptive_gate_enabled:
        adaptive_gate_info = compute_adaptive_keep_ratio_from_scores(
            [float(item["score"]) for item in segment_records],
            args,
        )
        if adaptive_gate_info.get("adaptive_skip_prune", False):
            return list(candidates), {
                "route_coarse_segment_gate": f"{mode}_adaptive_skipped",
                "coarse_segment_gate_before": len(candidates),
                "coarse_segment_gate_after": len(candidates),
                "coarse_segment_gate_num_segments": len(segments),
                "coarse_segment_gate_keep_segments": len(segments),
                "coarse_segment_gate_segment_size": segment_size,
                "coarse_segment_gate_min_keep": min_keep,
                "coarse_segment_gate_ratio": 1.0,
                "coarse_segment_gate_prune_ratio": 0.0,
                "coarse_segment_gate_ms": 1000.0 * (time.perf_counter() - start_time),
                "coarse_segment_gate_preview": segment_records[:12],
                "coarse_segment_gate_adaptive_enabled": True,
                "coarse_segment_gate_adaptive_keep_ratio": float(
                    adaptive_gate_info.get("adaptive_keep_ratio", 1.0)
                ),
                "coarse_segment_gate_adaptive_uncertainty": float(
                    adaptive_gate_info.get("adaptive_uncertainty", 1.0)
                ),
                "coarse_segment_gate_adaptive_reason": str(
                    adaptive_gate_info.get("adaptive_reason", "adaptive_skip")
                ),
            }

        target_keep_candidates = resolve_adaptive_keep_count(
            len(candidates),
            min_keep,
            max_keep,
            adaptive_gate_info,
        )

    if target_keep_candidates >= len(candidates):
        return list(candidates), {
            "route_coarse_segment_gate": (
                f"{mode}_adaptive_skipped_no_prune"
                if adaptive_gate_enabled
                else f"{mode}_skipped_no_prune"
            ),
            "coarse_segment_gate_before": len(candidates),
            "coarse_segment_gate_after": len(candidates),
            "coarse_segment_gate_num_segments": len(segments),
            "coarse_segment_gate_keep_segments": len(segments),
            "coarse_segment_gate_segment_size": segment_size,
            "coarse_segment_gate_min_keep": min_keep,
            "coarse_segment_gate_ratio": float(
                adaptive_gate_info.get("adaptive_keep_ratio", keep_ratio)
            ),
            "coarse_segment_gate_prune_ratio": 0.0,
            "coarse_segment_gate_ms": 1000.0 * (time.perf_counter() - start_time),
            "coarse_segment_gate_preview": segment_records[:12],
            "coarse_segment_gate_adaptive_enabled": bool(adaptive_gate_enabled),
            "coarse_segment_gate_adaptive_keep_ratio": float(
                adaptive_gate_info.get("adaptive_keep_ratio", 0.0)
            ),
            "coarse_segment_gate_adaptive_uncertainty": float(
                adaptive_gate_info.get("adaptive_uncertainty", 0.0)
            ),
            "coarse_segment_gate_adaptive_reason": str(
                adaptive_gate_info.get("adaptive_reason", "no_prune")
            ),
        }

    ranked_segments = sorted(
        segment_records,
        key=lambda item: (
            -float(item["score"]),
            int(item["start_t"]),
            int(item["segment_idx"]),
        ),
    )

    kept_segments = []
    kept_count = 0
    for segment in ranked_segments:
        kept_segments.append(segment)
        kept_count += int(segment["num_candidates"])
        if kept_count >= target_keep_candidates:
            break

    keep_ids = {
        int(candidate_id)
        for segment in kept_segments
        for candidate_id in segment["candidate_ids"]
    }
    gated_candidates = [
        cand for cand in candidates
        if int(cand["candidate_id"]) in keep_ids
    ]
    gated_candidates = sorted(
        gated_candidates,
        key=lambda cand: (int(cand["start_t"]), int(cand["candidate_id"])),
    )
    prune_ratio = 1.0 - (
        float(len(gated_candidates)) / max(1.0, float(len(candidates)))
    )
    return gated_candidates, {
        "route_coarse_segment_gate": mode,
        "coarse_segment_gate_before": len(candidates),
        "coarse_segment_gate_after": len(gated_candidates),
        "coarse_segment_gate_num_segments": len(segments),
        "coarse_segment_gate_keep_segments": len(kept_segments),
        "coarse_segment_gate_segment_size": segment_size,
        "coarse_segment_gate_min_keep": min_keep,
        "coarse_segment_gate_ratio": keep_ratio,
        "coarse_segment_gate_prune_ratio": float(prune_ratio),
        "coarse_segment_gate_ms": 1000.0 * (time.perf_counter() - start_time),
        "coarse_segment_gate_preview": kept_segments[:12],
        "coarse_segment_gate_adaptive_enabled": bool(adaptive_gate_enabled),
        "coarse_segment_gate_adaptive_keep_ratio": float(
            adaptive_gate_info.get("adaptive_keep_ratio", 0.0)
        ),
        "coarse_segment_gate_adaptive_uncertainty": float(
            adaptive_gate_info.get("adaptive_uncertainty", 0.0)
        ),
        "coarse_segment_gate_adaptive_reason": str(
            adaptive_gate_info.get("adaptive_reason", "fixed")
        ),
    }


def resolve_node_summary_gate_keep_count(args, effective_route_top_k, num_candidates):
    if num_candidates <= 0:
        return 0

    factor = max(1, int(getattr(args, "node_summary_gate_factor", 4)))
    min_keep = max(1, int(getattr(args, "node_summary_gate_min_keep", 24)))
    max_keep = int(getattr(args, "node_summary_gate_max_keep", 0))
    keep_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "node_summary_gate_keep_ratio", 0.0))),
    )

    keep_count = max(
        min_keep,
        int(effective_route_top_k) * factor,
        int(np.ceil(float(num_candidates) * keep_ratio)),
    )
    if max_keep > 0:
        keep_count = min(keep_count, max_keep)
    return min(max(1, keep_count), int(num_candidates))


def resolve_node_summary_adaptive_keep_count(
    args,
    effective_route_top_k,
    num_candidates,
    ranked_candidate_ids,
    summary_scores,
):
    if num_candidates <= 0:
        return 0, {
            "node_summary_gate_budget_mode": "adaptive",
            "node_summary_gate_budget_reason": "empty",
            "node_summary_gate_adaptive_gap": 0.0,
        }

    safety_factor = max(
        1.0,
        float(getattr(args, "node_summary_gate_adaptive_safety_factor", 4.0)),
    )
    min_keep = int(np.ceil(float(max(1, effective_route_top_k)) * safety_factor))
    min_keep = min(max(1, min_keep), int(num_candidates))
    max_keep = int(getattr(args, "node_summary_gate_max_keep", 0))
    if max_keep > 0:
        min_keep = min(min_keep, max_keep, int(num_candidates))

    if len(ranked_candidate_ids) <= min_keep:
        return len(ranked_candidate_ids), {
            "node_summary_gate_budget_mode": "adaptive",
            "node_summary_gate_budget_reason": "small_pool",
            "node_summary_gate_adaptive_min_keep": int(min_keep),
            "node_summary_gate_adaptive_gap": 0.0,
        }

    ranked_scores = np.asarray(
        [
            float(summary_scores.get(int(candidate_id), -1e9))
            for candidate_id in ranked_candidate_ids
        ],
        dtype=np.float32,
    )
    finite_mask = np.isfinite(ranked_scores)
    non_finite_count = int((~finite_mask).sum())
    ranked_scores = np.nan_to_num(
        ranked_scores,
        nan=-1.0e9,
        posinf=1.0e9,
        neginf=-1.0e9,
    )

    # Scores are sorted descending. A large positive drop means the query-sketch
    # relevance curve has a natural elbow; keep candidates before that elbow.
    drops = ranked_scores[:-1] - ranked_scores[1:]
    if drops.size == 0:
        chosen = len(ranked_candidate_ids)
        best_gap = 0.0
    else:
        search_start = max(0, int(min_keep) - 1)
        search_end = len(ranked_candidate_ids) - 1
        if max_keep > 0:
            search_end = min(search_end, max_keep - 1)
        if search_start >= search_end:
            chosen = min(max(int(min_keep), 1), len(ranked_candidate_ids))
            best_gap = 0.0
        else:
            window = drops[search_start:search_end]
            best_offset = int(np.argmax(window))
            best_idx = search_start + best_offset
            chosen = best_idx + 1
            best_gap = float(drops[best_idx])

    chosen = max(int(min_keep), int(chosen))
    if max_keep > 0:
        chosen = min(chosen, max_keep)
    chosen = min(max(1, chosen), len(ranked_candidate_ids))
    return chosen, {
        "node_summary_gate_budget_mode": "adaptive",
        "node_summary_gate_budget_reason": "max_score_gap",
        "node_summary_gate_adaptive_min_keep": int(min_keep),
        "node_summary_gate_adaptive_gap": float(best_gap),
        "node_summary_gate_adaptive_safety_factor": float(safety_factor),
        "node_summary_gate_non_finite_score_count": int(non_finite_count),
    }


def is_multi_prototype_summary_mode(mode):
    mode = str(mode or "mean_key").lower()
    return mode in {"multi_key", "mean_peak_boundary"}


def is_quest_minmax_summary_mode(mode):
    mode = str(mode or "mean_key").lower()
    return mode in {"quest_minmax", "minmax_key"}


def _gather_max_norm_key(stacked_keys):
    """Return one high-salience K prototype per layer/head."""
    if stacked_keys.shape[2] <= 1:
        return stacked_keys[:, :, :1, :]

    norms = stacked_keys.float().pow(2).sum(dim=-1)
    idx = norms.argmax(dim=2, keepdim=True).unsqueeze(-1)
    idx = idx.expand(-1, -1, 1, stacked_keys.shape[-1])
    return torch.gather(stacked_keys, dim=2, index=idx)


def build_candidate_key_summary(stacked_keys, mode="mean_key"):
    """Compress one candidate's full K block into a lightweight node sketch.

    Single-prototype modes return shape:
      (layers, kv_heads, 1, head_dim)

    Multi-prototype modes return shape:
      (layers, kv_heads, num_prototypes, head_dim)

    The prototype dimension is intentionally kept as the token dimension so the
    existing Q-K scorer can treat a sketch like a tiny KV block.
    """
    mode = str(mode or "mean_key").lower()
    if mode == "max_key":
        return torch.amax(stacked_keys, dim=2, keepdim=True)
    if mode == "mean_key":
        return stacked_keys.mean(dim=2, keepdim=True)
    if mode == "max_norm_key":
        return _gather_max_norm_key(stacked_keys)
    if mode in {"multi_key", "mean_peak_boundary"}:
        prototypes = [
            stacked_keys.mean(dim=2, keepdim=True),
            _gather_max_norm_key(stacked_keys),
            stacked_keys[:, :, :1, :],
            stacked_keys[:, :, -1:, :],
        ]
        return torch.cat(prototypes, dim=2).contiguous()
    if is_quest_minmax_summary_mode(mode):
        # This mode is scored by a dedicated Quest-style upper-bound path.
        # Keep the representation tensor-like for callers that only inspect
        # summary shape, but do not use it as ordinary K prototypes.
        return torch.stack(
            [
                torch.amin(stacked_keys, dim=2),
                torch.amax(stacked_keys, dim=2),
            ],
            dim=2,
        ).contiguous()
    raise ValueError(f"Unknown node_summary_gate_mode: {mode}")


def build_candidate_key_minmax_summary(stacked_keys):
    """Build Quest-style per-block key min/max metadata."""
    return {
        "k_min": torch.amin(stacked_keys, dim=2).contiguous(),
        "k_max": torch.amax(stacked_keys, dim=2).contiguous(),
    }


def _resolve_query_topk_for_summary(query_len, query_topk_ratio):
    query_len = int(query_len)
    if query_len <= 0:
        return 1
    try:
        ratio = float(query_topk_ratio)
    except (TypeError, ValueError):
        ratio = 0.25
    if ratio <= 0.0:
        ratio = 0.25
    return max(1, min(query_len, int(np.ceil(float(query_len) * ratio))))


def _pool_minmax_query_scores(query_scores, mode="mean", query_topk_ratio=0.25):
    """Pool Quest-style query-token scores into one score per head."""
    mode = str(mode or "mean").lower()
    if mode == "mean":
        return query_scores.mean(dim=-1).squeeze(0)
    if mode in {"query_mean_topk", "query_peak_topk"}:
        query_len = int(query_scores.shape[-1])
        topk = _resolve_query_topk_for_summary(query_len, query_topk_ratio)
        return query_scores.topk(k=topk, dim=-1).values.mean(dim=-1).squeeze(0)
    raise ValueError(f"Unknown qk token pooling mode for minmax summary: {mode}")


def score_minmax_key_summaries_from_query_cache(
    query_q_cache,
    minmax_summaries,
    model,
    layer_indices,
    qk_token_pooling="query_peak_topk",
    qk_query_topk_ratio=0.25,
):
    """Score candidate K_min/K_max sketches with query Q.

    This mirrors Quest's page upper-bound idea at Python prototype level:
    for each query dimension, choose K_max when q >= 0 and K_min otherwise,
    then sum the resulting optimistic dot product. The output shape matches
    the existing Q-K scorer: (num_candidates, num_heads).
    """
    if query_q_cache is None:
        raise ValueError("quest_minmax node summary requires query_q_cache")

    num_layers = int(query_q_cache.get("num_model_layers", len(model.model.layers)) or 0)
    normalized_layers = [
        int(lidx) if int(lidx) >= 0 else num_layers + int(lidx)
        for lidx in layer_indices
    ]
    candidate_scores = []

    for summary in minmax_summaries:
        per_layer_scores = []
        for layer_idx in normalized_layers:
            entry = query_q_cache["q_by_layer"][int(layer_idx)]
            query_q = entry["q"].float()
            num_heads = int(entry["num_heads"])
            num_kv_heads = int(entry["num_key_value_heads"])
            scale = float(entry["scale"])

            k_min = summary["k_min"][int(layer_idx)].to(query_q.device).float()
            k_max = summary["k_max"][int(layer_idx)].to(query_q.device).float()
            if num_kv_heads != num_heads:
                n_rep = num_heads // num_kv_heads
                k_min = k_min.repeat_interleave(n_rep, dim=0)
                k_max = k_max.repeat_interleave(n_rep, dim=0)

            optimistic_k = torch.where(
                query_q >= 0,
                k_max.unsqueeze(1),
                k_min.unsqueeze(1),
            )
            query_scores = (query_q * optimistic_k).sum(dim=-1) / scale
            per_head = _pool_minmax_query_scores(
                query_scores,
                mode=qk_token_pooling,
                query_topk_ratio=qk_query_topk_ratio,
            )
            per_layer_scores.append(per_head.detach().cpu().numpy())

        candidate_scores.append(np.mean(per_layer_scores, axis=0))

    return np.stack(candidate_scores, axis=0), {
        "layers": normalized_layers,
        "n_heads": int(candidate_scores[0].shape[0]) if candidate_scores else 0,
        "query_q_cache": True,
        "node_summary_scoring": "quest_minmax_upper_bound",
        "qk_token_pooling": str(qk_token_pooling),
        "qk_query_topk_ratio": float(qk_query_topk_ratio),
    }


def get_candidate_stacked_keys_for_summary(
    kv_3d,
    cand,
    split_kv,
    stack_keys_from_kv,
    candidate_key_cache=None,
    cache_enabled=False,
    populate_cache=False,
):
    candidate_id = int(cand["candidate_id"])
    cached_keys = None
    if cache_enabled and isinstance(candidate_key_cache, dict):
        cached_keys = candidate_key_cache.get(candidate_id)
    if cached_keys is not None:
        return cached_keys, True

    kv_chunk = split_kv(kv_3d, cand["start_t"], cand["end_t"])
    stacked_keys = stack_keys_from_kv(kv_chunk)
    if cache_enabled and populate_cache and isinstance(candidate_key_cache, dict):
        candidate_key_cache[candidate_id] = stacked_keys
    del kv_chunk
    return stacked_keys, False


def score_candidates_node_summary_gate(
    kv_3d,
    candidates,
    query_ids,
    model,
    scoring_layers,
    split_kv,
    qk_score_fn,
    qk_aggregation="mean",
    qk_topk=4,
    qk_token_pooling="mean",
    qk_query_topk_ratio=0.25,
    batch_size=64,
    summary_mode="mean_key",
    candidate_key_cache=None,
    cache_enabled=False,
    populate_key_cache=False,
    query_q_cache=None,
    cached_qk_score_fn=None,
):
    from distributed_sim import aggregate_qk_scores, stack_keys_from_kv

    if not candidates:
        return {}, {
            "node_summary_prepare_ms": 0.0,
            "node_summary_score_ms": 0.0,
            "node_summary_aggregate_ms": 0.0,
            "node_summary_cache_hits": 0,
            "node_summary_cache_misses": 0,
        }

    effective_batch_size = max(1, int(batch_size))
    summary_scores = {}
    cache_hits = 0
    cache_misses = 0
    non_finite_score_count = 0
    prepare_ms = 0.0
    score_ms = 0.0
    aggregate_ms = 0.0
    scoring_pooling = str(qk_token_pooling or "mean")
    if is_multi_prototype_summary_mode(summary_mode):
        scoring_pooling = "query_peak_topk"
    use_quest_minmax_summary = is_quest_minmax_summary_mode(summary_mode)
    if use_quest_minmax_summary:
        scoring_pooling = "query_peak_topk"

    for batch_start in range(0, len(candidates), effective_batch_size):
        batch_candidates = candidates[batch_start : batch_start + effective_batch_size]
        summary_keys_list = []
        prepare_start = time.perf_counter()
        for cand in batch_candidates:
            stacked_keys, cache_hit = get_candidate_stacked_keys_for_summary(
                kv_3d,
                cand,
                split_kv,
                stack_keys_from_kv,
                candidate_key_cache=candidate_key_cache,
                cache_enabled=cache_enabled,
                populate_cache=populate_key_cache,
            )
            if cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1
            if use_quest_minmax_summary:
                summary_keys_list.append(build_candidate_key_minmax_summary(stacked_keys))
            else:
                summary_keys_list.append(
                    build_candidate_key_summary(stacked_keys, mode=summary_mode)
                )
        prepare_ms += 1000.0 * (time.perf_counter() - prepare_start)

        score_start = time.perf_counter()
        if use_quest_minmax_summary:
            scores_matrix, _ = score_minmax_key_summaries_from_query_cache(
                query_q_cache,
                summary_keys_list,
                model,
                scoring_layers,
                qk_token_pooling=scoring_pooling,
                qk_query_topk_ratio=qk_query_topk_ratio,
            )
        elif query_q_cache is not None and cached_qk_score_fn is not None:
            scores_matrix, _ = cached_qk_score_fn(
                query_q_cache,
                summary_keys_list,
                model,
                scoring_layers,
                qk_token_pooling=scoring_pooling,
                qk_query_topk_ratio=qk_query_topk_ratio,
            )
        else:
            scores_matrix, _ = qk_score_fn(
                query_ids,
                summary_keys_list,
                model,
                scoring_layers,
                qk_token_pooling=scoring_pooling,
                qk_query_topk_ratio=qk_query_topk_ratio,
            )
        score_ms += 1000.0 * (time.perf_counter() - score_start)

        scores_matrix = np.asarray(scores_matrix, dtype=np.float32)
        if scores_matrix.ndim == 1:
            scores_matrix = scores_matrix.reshape(1, -1)

        aggregate_start = time.perf_counter()
        for cand, scores in zip(batch_candidates, scores_matrix):
            scalar = float(aggregate_qk_scores(scores, qk_aggregation, qk_topk))
            if not np.isfinite(scalar):
                non_finite_score_count += 1
                scalar = -1.0e9
            summary_scores[int(cand["candidate_id"])] = scalar
        aggregate_ms += 1000.0 * (time.perf_counter() - aggregate_start)
        del summary_keys_list, scores_matrix

    return summary_scores, {
        "node_summary_prepare_ms": float(prepare_ms),
        "node_summary_score_ms": float(score_ms),
        "node_summary_aggregate_ms": float(aggregate_ms),
        "node_summary_cache_hits": int(cache_hits),
        "node_summary_cache_misses": int(cache_misses),
        "node_summary_gate_non_finite_score_count": int(non_finite_score_count),
        "node_summary_gate_score_pooling": str(scoring_pooling),
    }


def apply_node_summary_gate(
    kv_3d,
    candidates,
    query_ids,
    model,
    scoring_layers,
    split_kv,
    qk_score_fn,
    args,
    effective_route_top_k,
    candidate_key_cache=None,
    query_q_cache=None,
    cached_qk_score_fn=None,
):
    mode = str(getattr(args, "node_summary_gate", "none") or "none").lower()
    summary_mode = str(
        getattr(args, "node_summary_gate_summary_mode", "mean_key") or "mean_key"
    ).lower()
    base_info = {
        "node_summary_gate": mode,
        "node_summary_gate_summary_mode": summary_mode,
        "node_summary_gate_before": int(len(candidates)),
        "node_summary_gate_after": int(len(candidates)),
        "node_summary_gate_target_keep": int(len(candidates)),
        "node_summary_gate_prune_ratio": 0.0,
        "node_summary_gate_keep_ratio": float(
            getattr(args, "node_summary_gate_keep_ratio", 0.0)
        ),
        "node_summary_gate_budget_mode": str(
            getattr(args, "node_summary_gate_budget_mode", "fixed") or "fixed"
        ),
        "node_summary_gate_budget_reason": "",
        "node_summary_gate_adaptive_min_keep": 0,
        "node_summary_gate_adaptive_gap": 0.0,
        "node_summary_gate_adaptive_safety_factor": float(
            getattr(args, "node_summary_gate_adaptive_safety_factor", 4.0)
        ),
        "node_summary_gate_non_finite_score_count": 0,
        "node_summary_gate_score_pooling": "",
        "node_summary_gate_min_keep": int(
            getattr(args, "node_summary_gate_min_keep", 24)
        ),
        "node_summary_gate_max_keep": int(
            getattr(args, "node_summary_gate_max_keep", 0)
        ),
        "node_summary_gate_factor": int(
            getattr(args, "node_summary_gate_factor", 4)
        ),
        "node_summary_gate_populate_key_cache": bool(
            getattr(args, "node_summary_gate_populate_key_cache", False)
        ),
        "node_summary_gate_ms": 0.0,
        "node_summary_prepare_ms": 0.0,
        "node_summary_score_ms": 0.0,
        "node_summary_aggregate_ms": 0.0,
        "node_summary_cache_hits": 0,
        "node_summary_cache_misses": 0,
        "node_summary_gate_preview": [],
        "candidate_ids_after_node_summary_gate": [
            int(cand["candidate_id"]) for cand in candidates
        ],
    }
    if mode == "none" or not candidates:
        return list(candidates), {}, base_info
    if mode != "qk_sketch":
        raise ValueError(f"Unknown node_summary_gate: {mode}")

    start_time = time.perf_counter()
    summary_scores, score_info = score_candidates_node_summary_gate(
        kv_3d,
        candidates,
        query_ids,
        model,
        scoring_layers,
        split_kv,
        qk_score_fn,
        qk_aggregation=getattr(args, "qk_aggregation", "mean"),
        qk_topk=getattr(args, "qk_topk", 4),
        qk_token_pooling=getattr(args, "qk_token_pooling", "mean"),
        qk_query_topk_ratio=getattr(args, "qk_query_topk_ratio", 0.25),
        batch_size=getattr(args, "node_summary_gate_batch_size", 64),
        summary_mode=summary_mode,
        candidate_key_cache=candidate_key_cache,
        cache_enabled=bool(getattr(args, "cache_candidate_keys", False)),
        populate_key_cache=bool(
            getattr(args, "node_summary_gate_populate_key_cache", False)
        ),
        query_q_cache=query_q_cache,
        cached_qk_score_fn=cached_qk_score_fn,
    )
    ranked_candidate_ids = sorted(
        [int(cand["candidate_id"]) for cand in candidates],
        key=lambda cid: (
            -float(
                np.nan_to_num(
                    summary_scores.get(int(cid), -1e9),
                    nan=-1.0e9,
                    posinf=1.0e9,
                    neginf=-1.0e9,
                )
            ),
            int(cid),
        ),
    )

    budget_mode = str(
        getattr(args, "node_summary_gate_budget_mode", "fixed") or "fixed"
    ).lower()
    if budget_mode == "score_only":
        target_keep = len(candidates)
        budget_info = {
            "node_summary_gate_budget_mode": "score_only",
            "node_summary_gate_budget_reason": "score_without_prune",
        }
    elif budget_mode == "adaptive":
        target_keep, budget_info = resolve_node_summary_adaptive_keep_count(
            args,
            effective_route_top_k,
            len(candidates),
            ranked_candidate_ids,
            summary_scores,
        )
    elif budget_mode == "fixed":
        target_keep = resolve_node_summary_gate_keep_count(
            args,
            effective_route_top_k,
            len(candidates),
        )
        budget_info = {
            "node_summary_gate_budget_mode": "fixed",
            "node_summary_gate_budget_reason": "fixed_count",
        }
    else:
        raise ValueError(f"Unknown node_summary_gate_budget_mode: {budget_mode}")

    base_info["node_summary_gate_target_keep"] = int(target_keep)
    base_info.update(budget_info)
    if target_keep >= len(candidates):
        returned_candidates = list(candidates)
        for cand in returned_candidates:
            cand["node_summary_score"] = float(
                summary_scores.get(int(cand["candidate_id"]), -1e9)
            )
        info = dict(base_info)
        info.update(score_info)
        info["node_summary_gate"] = f"{mode}_skipped_no_prune"
        info["node_summary_gate_ms"] = 1000.0 * (time.perf_counter() - start_time)
        info["node_summary_gate_preview"] = [
            {
                "candidate_id": int(candidate_id),
                "score": float(summary_scores.get(int(candidate_id), -1e9)),
            }
            for candidate_id in ranked_candidate_ids[:20]
        ]
        return returned_candidates, summary_scores, info

    keep_ids = set(int(cid) for cid in ranked_candidate_ids[:target_keep])
    gated_candidates = [
        cand for cand in candidates
        if int(cand["candidate_id"]) in keep_ids
    ]
    gated_candidates = sorted(
        gated_candidates,
        key=lambda cand: (int(cand["start_t"]), int(cand["candidate_id"])),
    )
    for cand in gated_candidates:
        cand["node_summary_score"] = float(
            summary_scores.get(int(cand["candidate_id"]), -1e9)
        )

    prune_ratio = 1.0 - (
        float(len(gated_candidates)) / max(1.0, float(len(candidates)))
    )
    preview = [
        {
            "candidate_id": int(candidate_id),
            "score": float(summary_scores.get(int(candidate_id), -1e9)),
        }
        for candidate_id in ranked_candidate_ids[:20]
    ]
    info = dict(base_info)
    info.update(score_info)
    info.update(budget_info)
    info.update(
        {
            "node_summary_gate_after": int(len(gated_candidates)),
            "node_summary_gate_prune_ratio": float(prune_ratio),
            "node_summary_gate_ms": 1000.0 * (time.perf_counter() - start_time),
            "node_summary_gate_preview": preview,
            "candidate_ids_after_node_summary_gate": [
                int(cand["candidate_id"]) for cand in gated_candidates
            ],
        }
    )
    return gated_candidates, summary_scores, info


def score_candidates_exact_qk_batched(
    kv_3d,
    candidates,
    query_ids,
    model,
    scoring_layers,
    split_kv,
    qk_score_fn,
    qk_aggregation="mean",
    qk_topk=4,
    qk_token_pooling="mean",
    qk_query_topk_ratio=0.25,
    batch_size=32,
    candidate_key_cache=None,
    cache_enabled=False,
    query_q_cache=None,
    cached_qk_score_fn=None,
    query_q_prepare_ms=0.0,
):
    from distributed_sim import aggregate_qk_scores, stack_keys_from_kv

    if not candidates:
        return [], [], {
            "candidate_key_cache_enabled": bool(cache_enabled),
            "candidate_key_cache_hits": 0,
            "candidate_key_cache_misses": 0,
            "candidate_key_prepare_ms": 0.0,
            "query_q_cache_enabled": bool(query_q_cache is not None),
            "query_q_prepare_ms": float(query_q_prepare_ms),
            "qk_model_inference_ms": 0.0,
            "qk_score_aggregation_ms": 0.0,
            "qk_total_stage_ms": 0.0,
            "qk_token_pooling": str(qk_token_pooling),
            "qk_query_topk_ratio": float(qk_query_topk_ratio),
            "candidate_key_cache_size": (
                len(candidate_key_cache) if isinstance(candidate_key_cache, dict) else 0
            ),
        }

    effective_batch_size = max(1, int(batch_size))
    scored_candidates = []
    per_head_scores = []
    cache_hits = 0
    cache_misses = 0
    candidate_key_prepare_ms = 0.0
    qk_model_inference_ms = 0.0
    qk_score_aggregation_ms = 0.0

    for batch_start in range(0, len(candidates), effective_batch_size):
        batch_candidates = candidates[batch_start : batch_start + effective_batch_size]
        chunk_keys_list = []
        candidate_key_prepare_start_time = time.perf_counter()

        for cand in batch_candidates:
            candidate_id = int(cand["candidate_id"])
            cached_keys = None
            if cache_enabled and isinstance(candidate_key_cache, dict):
                cached_keys = candidate_key_cache.get(candidate_id)

            if cached_keys is not None:
                chunk_keys_list.append(cached_keys)
                cache_hits += 1
                continue

            kv_chunk = split_kv(kv_3d, cand["start_t"], cand["end_t"])
            stacked_keys = stack_keys_from_kv(kv_chunk)
            if cache_enabled and isinstance(candidate_key_cache, dict):
                candidate_key_cache[candidate_id] = stacked_keys
            chunk_keys_list.append(stacked_keys)
            cache_misses += 1
            del kv_chunk
        candidate_key_prepare_ms += 1000.0 * (
            time.perf_counter() - candidate_key_prepare_start_time
        )

        qk_model_inference_start_time = time.perf_counter()
        if query_q_cache is not None and cached_qk_score_fn is not None:
            scores_matrix, _ = cached_qk_score_fn(
                query_q_cache,
                chunk_keys_list,
                model,
                scoring_layers,
                qk_token_pooling=qk_token_pooling,
                qk_query_topk_ratio=qk_query_topk_ratio,
            )
        else:
            scores_matrix, _ = qk_score_fn(
                query_ids,
                chunk_keys_list,
                model,
                scoring_layers,
                qk_token_pooling=qk_token_pooling,
                qk_query_topk_ratio=qk_query_topk_ratio,
            )
        qk_model_inference_ms += 1000.0 * (
            time.perf_counter() - qk_model_inference_start_time
        )
        scores_matrix = np.asarray(scores_matrix, dtype=np.float32)
        if scores_matrix.ndim == 1:
            scores_matrix = scores_matrix.reshape(1, -1)

        qk_score_aggregation_start_time = time.perf_counter()
        for cand, scores in zip(batch_candidates, scores_matrix):
            scalar = aggregate_qk_scores(scores, qk_aggregation, qk_topk)
            record = dict(cand)
            record["score"] = float(scalar)
            scored_candidates.append(record)
            per_head_scores.append(np.asarray(scores, dtype=np.float32).reshape(-1))
        qk_score_aggregation_ms += 1000.0 * (
            time.perf_counter() - qk_score_aggregation_start_time
        )

        del chunk_keys_list, scores_matrix

    qk_total_stage_ms = (
        float(candidate_key_prepare_ms)
        + float(query_q_prepare_ms)
        + float(qk_model_inference_ms)
        + float(qk_score_aggregation_ms)
    )
    cache_stats = {
        "candidate_key_cache_enabled": bool(cache_enabled),
        "candidate_key_cache_hits": int(cache_hits),
        "candidate_key_cache_misses": int(cache_misses),
        "candidate_key_prepare_ms": float(candidate_key_prepare_ms),
        "query_q_cache_enabled": bool(query_q_cache is not None),
        "query_q_prepare_ms": float(query_q_prepare_ms),
        "qk_model_inference_ms": float(qk_model_inference_ms),
        "qk_score_aggregation_ms": float(qk_score_aggregation_ms),
        "qk_total_stage_ms": float(qk_total_stage_ms),
        "qk_token_pooling": str(qk_token_pooling),
        "qk_query_topk_ratio": float(qk_query_topk_ratio),
        "candidate_key_cache_size": (
            len(candidate_key_cache) if isinstance(candidate_key_cache, dict) else 0
        ),
    }
    return scored_candidates, per_head_scores, cache_stats


def build_mainline_doc_routing_artifacts(
    meeting,
    transcripts,
    turn_boundaries,
    total_tokens,
    args,
    topic_index=None,
):
    if topic_index is not None:
        topic_nodes = topic_index["topic_nodes"]
        turn_to_topic_ids = topic_index["turn_to_topic_ids"]
        topic_turn_info = topic_index["topic_turn_info"]
    else:
        topic_nodes, turn_to_topic_ids = build_topic_nodes(meeting, num_turns=len(transcripts))
        topic_turn_info = {topic["topic_id"]: topic["turns"] for topic in topic_nodes}

    topic_to_virtual_node_id, virtual_node_to_topic_ids = assign_topics_to_virtual_nodes(
        topic_nodes,
        getattr(args, "num_nodes", 1),
        getattr(args, "node_assignment_mode", "contiguous"),
        getattr(args, "topic_node_layout_path", None),
    )
    route_candidates = build_hierarchical_candidates(
        turn_boundaries,
        total_tokens,
        turn_to_topic_ids,
        max(1, args.route_chunk_size),
    )
    lexical_stats = build_topic_lexical_stats(
        topic_nodes,
        transcripts,
        topic_index=topic_index,
        label_repeat=args.lexical_label_repeat,
    )
    candidate_lexical_stats = build_candidate_lexical_stats(
        route_candidates,
        transcripts,
        topic_nodes,
    )

    return {
        "topic_nodes": topic_nodes,
        "turn_to_topic_ids": turn_to_topic_ids,
        "topic_turn_info": topic_turn_info,
        "topic_to_virtual_node_id": topic_to_virtual_node_id,
        "virtual_node_to_topic_ids": virtual_node_to_topic_ids,
        "route_candidates": route_candidates,
        "lexical_stats": lexical_stats,
        "candidate_lexical_stats": candidate_lexical_stats,
        "candidate_key_cache": {},
    }


def bm25_score_tokens(
    query_tokens,
    doc_tf,
    doc_len,
    avg_doc_len,
    idf_map,
    k1=1.2,
    b=0.75,
):
    if not query_tokens or not doc_tf:
        return -1e9

    score = 0.0
    denom_base = k1 * (1.0 - b + b * (float(doc_len) / max(float(avg_doc_len), 1.0)))
    for tok in query_tokens:
        tf = float(doc_tf.get(tok, 0.0))
        if tf <= 0.0:
            continue
        idf = float(idf_map.get(tok, 0.0))
        score += idf * ((tf * (k1 + 1.0)) / (tf + denom_base))
    return float(score) if score > 0.0 else -1e9


def score_topics_lexical_qmsum(
    query_text,
    topic_nodes,
    transcripts,
    topic_index=None,
    label_repeat=3,
    lexical_stats=None,
):
    query_tokens = tokenize_lexical_text(query_text)
    if not query_tokens:
        return {int(topic["topic_id"]): -1e9 for topic in topic_nodes}

    if lexical_stats is None:
        lexical_stats = build_topic_lexical_stats(
            topic_nodes,
            transcripts,
            topic_index=topic_index,
            label_repeat=label_repeat,
        )
    scores = {}
    for topic in topic_nodes:
        topic_id = int(topic["topic_id"])
        score = bm25_score_tokens(
            query_tokens,
            lexical_stats["doc_tf"].get(topic_id, {}),
            lexical_stats["doc_lens"].get(topic_id, 0),
            lexical_stats["avg_doc_len"],
            lexical_stats["idf"],
        )
        if topic.get("is_gap", False) and score > -1e8:
            score -= 0.05
        scores[topic_id] = float(score)
    return scores


def rerank_topics_with_full_lexical_qmsum(
    query_text,
    candidate_topic_ids,
    topic_nodes,
    transcripts,
    label_repeat=3,
):
    candidate_set = set(int(x) for x in candidate_topic_ids)
    candidate_topics = [topic for topic in topic_nodes if int(topic["topic_id"]) in candidate_set]
    if not candidate_topics:
        return {}

    query_tokens = tokenize_lexical_text(query_text)
    if not query_tokens:
        return {int(topic["topic_id"]): -1e9 for topic in candidate_topics}

    doc_tf = {}
    doc_lens = {}
    doc_freq = defaultdict(int)

    for topic in candidate_topics:
        topic_id = int(topic["topic_id"])
        tokens = []

        label = topic.get("label", "").strip()
        if label:
            tokens.extend(tokenize_lexical_text(label) * max(1, int(label_repeat)))

        for turn_idx in topic.get("turns", []):
            if 0 <= int(turn_idx) < len(transcripts):
                turn = transcripts[int(turn_idx)]
                speaker = turn.get("speaker", "").strip() or "Speaker"
                content = turn.get("content", "").strip()
                tokens.extend(tokenize_lexical_text(f"{speaker}: {content}"))

        tf = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        doc_tf[topic_id] = dict(tf)
        doc_lens[topic_id] = len(tokens)
        for tok in tf.keys():
            doc_freq[tok] += 1

    avg_doc_len = float(np.mean(list(doc_lens.values()))) if doc_lens else 0.0
    num_docs = max(1, len(candidate_topics))
    idf = {}
    for tok, freq in doc_freq.items():
        idf[tok] = float(np.log(1.0 + ((num_docs - freq + 0.5) / (freq + 0.5))))

    scores = {}
    for topic in candidate_topics:
        topic_id = int(topic["topic_id"])
        score = bm25_score_tokens(
            query_tokens,
            doc_tf.get(topic_id, {}),
            doc_lens.get(topic_id, 0),
            avg_doc_len,
            idf,
        )
        if topic.get("is_gap", False) and score > -1e8:
            score -= 0.05
        scores[topic_id] = float(score)
    return scores


def score_mainline_topic_chunk(
    kv_3d,
    turn_boundaries,
    total_tokens,
    query_text,
    query_ids,
    model,
    scoring_layers,
    split_kv,
    qk_score_fn,
    meeting,
    transcripts,
    args,
    topic_index=None,
    routing_artifacts=None,
):
    from distributed_sim import rank_scores

    if routing_artifacts is not None:
        topic_nodes = routing_artifacts["topic_nodes"]
        turn_to_topic_ids = routing_artifacts["turn_to_topic_ids"]
        topic_turn_info = routing_artifacts["topic_turn_info"]
        topic_to_virtual_node_id = routing_artifacts["topic_to_virtual_node_id"]
        virtual_node_to_topic_ids = routing_artifacts["virtual_node_to_topic_ids"]
        route_candidates = routing_artifacts["route_candidates"]
        lexical_stats = routing_artifacts.get("lexical_stats")
        candidate_lexical_stats = routing_artifacts.get("candidate_lexical_stats")
        candidate_key_cache = routing_artifacts.get("candidate_key_cache")
    elif topic_index is not None:
        topic_nodes = topic_index["topic_nodes"]
        turn_to_topic_ids = topic_index["turn_to_topic_ids"]
        topic_turn_info = topic_index["topic_turn_info"]
        topic_to_virtual_node_id, virtual_node_to_topic_ids = assign_topics_to_virtual_nodes(
            topic_nodes,
            getattr(args, "num_nodes", 1),
            getattr(args, "node_assignment_mode", "contiguous"),
            getattr(args, "topic_node_layout_path", None),
        )
        route_candidates = build_hierarchical_candidates(
            turn_boundaries,
            total_tokens,
            turn_to_topic_ids,
            max(1, args.route_chunk_size),
        )
        lexical_stats = None
        candidate_lexical_stats = build_candidate_lexical_stats(
            route_candidates,
            transcripts,
            topic_nodes,
        )
        candidate_key_cache = None
    else:
        topic_nodes, turn_to_topic_ids = build_topic_nodes(meeting, num_turns=len(transcripts))
        topic_turn_info = {topic["topic_id"]: topic["turns"] for topic in topic_nodes}
        topic_to_virtual_node_id, virtual_node_to_topic_ids = assign_topics_to_virtual_nodes(
            topic_nodes,
            getattr(args, "num_nodes", 1),
            getattr(args, "node_assignment_mode", "contiguous"),
            getattr(args, "topic_node_layout_path", None),
        )
        route_candidates = build_hierarchical_candidates(
            turn_boundaries,
            total_tokens,
            turn_to_topic_ids,
            max(1, args.route_chunk_size),
        )
        lexical_stats = None
        candidate_lexical_stats = build_candidate_lexical_stats(
            route_candidates,
            transcripts,
            topic_nodes,
        )
        candidate_key_cache = None

    coarse_topic_routing_start_time = time.perf_counter()
    lexical_scores = score_topics_lexical_qmsum(
        query_text,
        topic_nodes,
        transcripts,
        topic_index=topic_index,
        label_repeat=args.lexical_label_repeat,
        lexical_stats=lexical_stats,
    )
    lexical_result = rank_scores(lexical_scores)
    lexical_ranked_topics = list(lexical_result["ranked_nodes"])
    lexical_debug = {
        "prototype_ranked_topics": [int(x) for x in lexical_ranked_topics],
        "prototype_scores": {int(k): float(v) for k, v in lexical_scores.items()},
        "rerank_mode": "none",
    }

    if args.coarse_lexical_rerank_topk > 1 and lexical_ranked_topics:
        rerank_topk = min(
            max(1, int(args.coarse_lexical_rerank_topk)),
            len(lexical_ranked_topics),
        )
        rerank_candidates = lexical_ranked_topics[:rerank_topk]
        full_lexical_scores = rerank_topics_with_full_lexical_qmsum(
            query_text,
            rerank_candidates,
            topic_nodes,
            transcripts,
            label_repeat=args.lexical_label_repeat,
        )
        if full_lexical_scores:
            rerank_result = rank_scores(full_lexical_scores)
            reranked_prefix = list(rerank_result["ranked_nodes"])
            remaining = [x for x in lexical_ranked_topics if x not in set(reranked_prefix)]
            lexical_ranked_topics = reranked_prefix + remaining
            lexical_result = {
                "top_node": int(lexical_ranked_topics[0]),
                "ranked_nodes": lexical_ranked_topics,
                "scores": {
                    int(topic_id): float(
                        full_lexical_scores.get(topic_id, lexical_scores.get(topic_id, -1e9))
                    )
                    for topic_id in lexical_ranked_topics
                },
            }
            lexical_debug["rerank_mode"] = "full_topic_lexical"
            lexical_debug["rerank_candidates"] = [int(x) for x in rerank_candidates]
            lexical_debug["full_topic_scores"] = {
                int(k): float(v) for k, v in full_lexical_scores.items()
            }

    selected_topic_ids, adaptive_topic_rescue_info = maybe_apply_adaptive_topic_rescue(
        lexical_ranked_topics,
        lexical_result,
        topic_nodes,
        args,
    )
    k_topics = len(selected_topic_ids)
    lexical_debug["adaptive_topic_rescue"] = adaptive_topic_rescue_info
    selected_topic_set = set(int(x) for x in selected_topic_ids)
    candidate_topic_scope = str(
        getattr(args, "candidate_topic_scope", "selected_topics") or "selected_topics"
    ).lower()
    if candidate_topic_scope not in {"selected_topics", "all_topics"}:
        raise ValueError(f"Unknown candidate_topic_scope: {candidate_topic_scope}")
    coarse_topic_routing_ms = 1000.0 * (
        time.perf_counter() - coarse_topic_routing_start_time
    )

    topic_filter_start_time = time.perf_counter()
    if candidate_topic_scope == "all_topics":
        candidates_to_score = list(route_candidates)
    else:
        candidates_to_score = [
            cand
            for cand in route_candidates
            if any(int(topic_id) in selected_topic_set for topic_id in cand["topic_ids"])
        ]
    topic_filter_ms = 1000.0 * (time.perf_counter() - topic_filter_start_time)
    candidate_ids_after_topic_filter = [
        int(cand["candidate_id"]) for cand in candidates_to_score
    ]

    effective_route_top_k, query_budget_type, route_budget_mode = resolve_route_top_k_for_query(
        args,
        query_text,
    )
    candidate_prefilter_start_time = time.perf_counter()
    candidate_prefilter_requested_mode = str(
        getattr(args, "route_candidate_prefilter", "none")
    ).lower()
    candidate_prefilter_mode = candidate_prefilter_requested_mode
    candidate_prefilter_min_prune_ratio = resolve_candidate_prefilter_min_prune_ratio(args)
    num_candidates_before_prefilter = len(candidates_to_score)
    candidate_prefilter_pool_size = len(candidates_to_score)
    candidate_prefilter_requested_pool_size = len(candidates_to_score)
    candidate_prefilter_prune_ratio = 0.0
    candidate_prefilter_skip_reason = ""
    candidate_prefilter_scores = {}
    candidate_prefilter_selected_ids = []
    candidate_prefilter_adaptive_info = {}

    if candidate_prefilter_requested_mode == "lexical" and candidates_to_score:
        candidate_ids = [int(cand["candidate_id"]) for cand in candidates_to_score]
        if bool(getattr(args, "route_adaptive_prefilter", False)):
            candidate_prefilter_scores = score_candidates_lexical_prefilter(
                query_text,
                candidate_ids,
                candidate_lexical_stats,
            )
            candidate_prefilter_adaptive_info = compute_adaptive_keep_ratio_from_scores(
                [
                    candidate_prefilter_scores.get(int(candidate_id), -1e9)
                    for candidate_id in candidate_ids
                ],
                args,
            )
            adaptive_min_keep = resolve_candidate_prefilter_fixed_floor(
                args,
                effective_route_top_k,
                len(candidates_to_score),
            )
            candidate_prefilter_requested_pool_size = resolve_adaptive_keep_count(
                len(candidates_to_score),
                adaptive_min_keep,
                int(getattr(args, "route_candidate_prefilter_max_keep", 0)),
                candidate_prefilter_adaptive_info,
            )
        else:
            candidate_prefilter_requested_pool_size = resolve_candidate_prefilter_size(
                args,
                effective_route_top_k,
                len(candidates_to_score),
            )
        candidate_prefilter_prune_ratio = 1.0 - (
            float(candidate_prefilter_requested_pool_size)
            / max(1.0, float(num_candidates_before_prefilter))
        )
        if candidate_prefilter_adaptive_info.get("adaptive_skip_prune", False):
            candidate_prefilter_mode = "lexical_adaptive_skipped"
            candidate_prefilter_pool_size = len(candidates_to_score)
            candidate_prefilter_requested_pool_size = len(candidates_to_score)
            candidate_prefilter_prune_ratio = 0.0
            candidate_prefilter_skip_reason = str(
                candidate_prefilter_adaptive_info.get("adaptive_reason", "adaptive_skip")
            )
            for cand in candidates_to_score:
                cand["prefilter_score"] = float(
                    candidate_prefilter_scores.get(int(cand["candidate_id"]), -1e9)
                )
        elif candidate_prefilter_prune_ratio < candidate_prefilter_min_prune_ratio:
            candidate_prefilter_mode = "lexical_skipped_low_prune"
            candidate_prefilter_pool_size = len(candidates_to_score)
            candidate_prefilter_skip_reason = (
                "requested pool would prune too few candidates"
            )
            for cand in candidates_to_score:
                cand["prefilter_score"] = float(
                    candidate_prefilter_scores.get(int(cand["candidate_id"]), -1e9)
                )
        else:
            if bool(getattr(args, "route_adaptive_prefilter", False)):
                candidate_prefilter_mode = "lexical_adaptive"
            candidate_prefilter_pool_size = candidate_prefilter_requested_pool_size
            if not candidate_prefilter_scores:
                candidate_prefilter_scores = score_candidates_lexical_prefilter(
                    query_text,
                    candidate_ids,
                    candidate_lexical_stats,
                )
            ranked_candidate_ids = sorted(
                candidate_ids,
                key=lambda cid: (
                    -float(candidate_prefilter_scores.get(int(cid), -1e9)),
                    int(cid),
                ),
            )
            candidate_prefilter_selected_ids = ranked_candidate_ids[:candidate_prefilter_pool_size]
            keep_set = set(int(cid) for cid in candidate_prefilter_selected_ids)
            candidates_to_score = [
                cand for cand in candidates_to_score
                if int(cand["candidate_id"]) in keep_set
            ]
            for cand in candidates_to_score:
                cand["prefilter_score"] = float(
                    candidate_prefilter_scores.get(int(cand["candidate_id"]), -1e9)
                )
    else:
        if candidate_prefilter_requested_mode != "none" and not candidates_to_score:
            candidate_prefilter_mode = f"{candidate_prefilter_requested_mode}_skipped_empty"
            candidate_prefilter_skip_reason = "no candidates after topic filter"
        for cand in candidates_to_score:
            cand["prefilter_score"] = -1e9
    candidate_prefilter_ms = 1000.0 * (time.perf_counter() - candidate_prefilter_start_time)
    num_candidates_after_prefilter = len(candidates_to_score)
    candidate_ids_after_prefilter = [
        int(cand["candidate_id"]) for cand in candidates_to_score
    ]

    dynamic_candidate_pool_start_time = time.perf_counter()
    num_candidates_before_dynamic_pool = len(candidates_to_score)
    dynamic_candidate_pool_target = num_candidates_before_dynamic_pool
    dynamic_candidate_pool_reason = "disabled"
    dynamic_candidate_pool_prune_ratio = 0.0
    dynamic_candidate_pool_selected_ids = []
    if candidates_to_score:
        (
            dynamic_candidate_pool_target,
            dynamic_candidate_pool_reason,
            dynamic_candidate_pool_prune_ratio,
        ) = resolve_candidate_pool_budget(
            args,
            query_budget_type,
            effective_route_top_k,
            num_candidates_before_dynamic_pool,
        )
        if dynamic_candidate_pool_target < num_candidates_before_dynamic_pool:
            if not candidate_prefilter_scores:
                candidate_ids = [
                    int(cand["candidate_id"]) for cand in candidates_to_score
                ]
                candidate_prefilter_scores = score_candidates_lexical_prefilter(
                    query_text,
                    candidate_ids,
                    candidate_lexical_stats,
                )
                for cand in candidates_to_score:
                    cand["prefilter_score"] = float(
                        candidate_prefilter_scores.get(
                            int(cand["candidate_id"]), -1e9
                        )
                    )
            ranked_dynamic_ids = sorted(
                [int(cand["candidate_id"]) for cand in candidates_to_score],
                key=lambda cid: (
                    -float(candidate_prefilter_scores.get(int(cid), -1e9)),
                    int(cid),
                ),
            )
            dynamic_candidate_pool_selected_ids = ranked_dynamic_ids[
                :dynamic_candidate_pool_target
            ]
            keep_set = set(int(cid) for cid in dynamic_candidate_pool_selected_ids)
            candidates_to_score = [
                cand for cand in candidates_to_score
                if int(cand["candidate_id"]) in keep_set
            ]
    dynamic_candidate_pool_ms = 1000.0 * (
        time.perf_counter() - dynamic_candidate_pool_start_time
    )
    num_candidates_after_dynamic_pool = len(candidates_to_score)
    candidate_ids_after_dynamic_pool = [
        int(cand["candidate_id"]) for cand in candidates_to_score
    ]

    candidates_to_score, coarse_segment_gate_info = apply_coarse_segment_gate(
        candidates_to_score,
        candidate_prefilter_scores,
        args,
    )
    num_candidates_after_coarse_segment_gate = len(candidates_to_score)
    candidate_ids_after_coarse_segment_gate = [
        int(cand["candidate_id"]) for cand in candidates_to_score
    ]

    topic_scores = {int(topic["topic_id"]): -1e9 for topic in topic_nodes}
    topic_score_buckets = defaultdict(list)

    query_q_cache = None
    cached_qk_score_fn = None
    query_q_prepare_ms = 0.0
    needs_query_q_cache = bool(getattr(args, "cache_query_q", False)) or (
        str(getattr(args, "node_summary_gate", "none") or "none").lower()
        == "qk_sketch"
    )
    if needs_query_q_cache and candidates_to_score:
        from experiment_chunk_split import (
            _chunk_qk_scores_per_head_cached,
            build_query_q_cache,
        )

        query_q_prepare_start_time = time.perf_counter()
        query_q_cache = build_query_q_cache(query_ids, model, scoring_layers)
        query_q_prepare_ms = 1000.0 * (
            time.perf_counter() - query_q_prepare_start_time
        )
        cached_qk_score_fn = _chunk_qk_scores_per_head_cached

    candidates_to_score, node_summary_scores, node_summary_gate_info = apply_node_summary_gate(
        kv_3d,
        candidates_to_score,
        query_ids,
        model,
        scoring_layers,
        split_kv,
        qk_score_fn,
        args,
        effective_route_top_k,
        candidate_key_cache=candidate_key_cache,
        query_q_cache=query_q_cache,
        cached_qk_score_fn=cached_qk_score_fn,
    )
    num_candidates_after_node_summary_gate = len(candidates_to_score)
    candidate_ids_after_node_summary_gate = [
        int(cand["candidate_id"]) for cand in candidates_to_score
    ]

    scored_candidates, per_head_scores, candidate_key_cache_stats = score_candidates_exact_qk_batched(
        kv_3d,
        candidates_to_score,
        query_ids,
        model,
        scoring_layers,
        split_kv,
        qk_score_fn,
        qk_aggregation=args.qk_aggregation,
        qk_topk=args.qk_topk,
        qk_token_pooling=getattr(args, "qk_token_pooling", "mean"),
        qk_query_topk_ratio=getattr(args, "qk_query_topk_ratio", 0.25),
        batch_size=getattr(args, "qk_score_batch_size", 32),
        candidate_key_cache=candidate_key_cache,
        cache_enabled=bool(getattr(args, "cache_candidate_keys", False)),
        query_q_cache=query_q_cache,
        cached_qk_score_fn=cached_qk_score_fn,
        query_q_prepare_ms=query_q_prepare_ms,
    )
    for record in scored_candidates:
        for topic_id in record["topic_ids"]:
            topic_id = int(topic_id)
            if candidate_topic_scope == "all_topics" or topic_id in selected_topic_set:
                topic_score_buckets[topic_id].append(float(record["score"]))
    qk_scoring_ms = float(
        candidate_key_cache_stats.get("query_q_prepare_ms", 0.0)
        + candidate_key_cache_stats.get("qk_model_inference_ms", 0.0)
        + candidate_key_cache_stats.get("qk_score_aggregation_ms", 0.0)
    )

    selection_postprocess_start_time = time.perf_counter()
    for topic_id, bucket in topic_score_buckets.items():
        if args.hier_topic_score_mode == "sum":
            topic_scores[topic_id] = float(sum(bucket))
        elif args.hier_topic_score_mode == "max":
            topic_scores[topic_id] = float(max(bucket))
        elif args.hier_topic_score_mode == "topk_mean":
            topk = min(max(1, args.hier_topic_topk), len(bucket))
            topic_scores[topic_id] = float(np.mean(sorted(bucket, reverse=True)[:topk]))
        else:
            raise ValueError(f"Unknown hier_topic_score_mode: {args.hier_topic_score_mode}")

    qk_result = rank_scores(topic_scores)
    strategy_results = {
        "lexical": lexical_result,
        "qk": qk_result,
    }

    selection_mode = str(getattr(args, "route_selection_mode", "chunk_topk")).lower()
    if selection_mode == "turn_rerank":
        selected_candidates, route_selection_debug = select_candidates_turn_rerank(
            scored_candidates,
            per_head_scores,
            candidate_prefilter_scores,
            transcripts,
            query_text,
            effective_route_top_k,
            args,
        )
    elif selection_mode == "turn_utility":
        selected_candidates, route_selection_debug = select_candidates_turn_utility(
            scored_candidates,
            per_head_scores,
            candidate_prefilter_scores,
            effective_route_top_k,
            args,
        )
    elif selection_mode == "turn_rank_fusion":
        selected_candidates, route_selection_debug = select_candidates_turn_rank_fusion(
            scored_candidates,
            per_head_scores,
            candidate_prefilter_scores,
            effective_route_top_k,
            args,
        )
    elif selection_mode == "hybrid":
        selected_candidates, route_selection_debug = select_candidates_hybrid(
            scored_candidates,
            per_head_scores,
            candidate_prefilter_scores,
            transcripts,
            query_text,
            effective_route_top_k,
            args,
        )
    elif selection_mode == "turn_unique":
        selected_candidates, route_selection_debug = select_candidates_turn_unique(
            scored_candidates,
            per_head_scores,
            effective_route_top_k,
            bool(args.route_per_head),
            args,
        )
    elif selection_mode == "turn_unique_guard":
        selected_candidates, route_selection_debug = select_candidates_turn_unique_guard(
            scored_candidates,
            per_head_scores,
            effective_route_top_k,
            bool(args.route_per_head),
            args,
        )
    elif selection_mode == "turn_unique_soft":
        selected_candidates, route_selection_debug = select_candidates_turn_unique_soft(
            scored_candidates,
            per_head_scores,
            effective_route_top_k,
            bool(args.route_per_head),
            args,
        )
    elif selection_mode == "topic_balanced":
        selected_candidates, route_selection_debug = select_candidates_topic_balanced(
            scored_candidates,
            per_head_scores,
            effective_route_top_k,
            bool(args.route_per_head),
            selected_topic_ids,
            args,
        )
    elif selection_mode == "topic_soft_rescue":
        selected_candidates, route_selection_debug = select_candidates_topic_soft_rescue(
            scored_candidates,
            per_head_scores,
            effective_route_top_k,
            bool(args.route_per_head),
            selected_topic_ids,
            args,
        )
    elif selection_mode == "evidence_pack":
        selected_candidates, route_selection_debug = select_candidates_evidence_pack(
            scored_candidates,
            per_head_scores,
            candidate_prefilter_scores,
            transcripts,
            query_text,
            effective_route_top_k,
            args,
        )
    elif selection_mode == "evidence_pack_v2":
        selected_candidates, route_selection_debug = select_candidates_evidence_pack_v2(
            scored_candidates,
            per_head_scores,
            candidate_prefilter_scores,
            transcripts,
            query_text,
            effective_route_top_k,
            args,
        )
    elif selection_mode == "evidence_pack_v3":
        selected_candidates, route_selection_debug = select_candidates_evidence_pack_v3(
            scored_candidates,
            per_head_scores,
            candidate_prefilter_scores,
            transcripts,
            query_text,
            effective_route_top_k,
            args,
        )
    else:
        selection_mode = "chunk_topk"
        selected_candidates, route_selection_debug = select_candidates_chunk_topk(
            scored_candidates,
            per_head_scores,
            effective_route_top_k,
            bool(args.route_per_head),
        )

    if args.route_neighbor_expand > 0 and selected_candidates:
        candidate_lookup = {
            (int(c["turn_idx"]), int(c["local_chunk_idx"])): c
            for c in scored_candidates
        }
        expanded_candidates = list(selected_candidates)
        expanded_keys = {
            (int(c["turn_idx"]), int(c["local_chunk_idx"])) for c in selected_candidates
        }

        for cand in list(selected_candidates):
            turn_idx = int(cand["turn_idx"])
            chunk_idx = int(cand["local_chunk_idx"])
            for delta in range(1, args.route_neighbor_expand + 1):
                for neighbor_chunk_idx in (chunk_idx - delta, chunk_idx + delta):
                    key = (turn_idx, neighbor_chunk_idx)
                    if key in expanded_keys:
                        continue
                    neighbor = candidate_lookup.get(key)
                    if neighbor is None:
                        continue
                    expanded_candidates.append(neighbor)
                    expanded_keys.add(key)

        selected_candidates = sorted(expanded_candidates, key=lambda x: -x["score"])
        route_selection_debug["neighbor_expand"] = int(args.route_neighbor_expand)
        route_selection_debug["selected_after_neighbor_expand"] = int(
            len(selected_candidates)
        )

    if args.route_diversity_filter and selected_candidates:
        diversity_target_keep = max(
            args.route_diversity_min_keep,
            int(np.ceil(len(selected_candidates) * args.route_diversity_keep_ratio)),
        )
        selected_candidates = diversify_selected_candidates(
            selected_candidates,
            transcripts,
            target_keep_count=diversity_target_keep,
            min_keep_count=args.route_diversity_min_keep,
            max_similarity=args.route_diversity_max_similarity,
        )
        selected_candidates = sorted(selected_candidates, key=lambda x: -x["score"])
        route_selection_debug["diversity_filter_applied"] = True
        route_selection_debug["selected_after_diversity"] = int(len(selected_candidates))

    qk_ranked_candidates = sorted(
        scored_candidates,
        key=lambda x: (-float(x.get("score", 0.0)), int(x["candidate_id"])),
    )
    qk_ranked_candidate_ids = [
        int(cand["candidate_id"]) for cand in qk_ranked_candidates
    ]
    selected_candidate_ids = [
        int(cand["candidate_id"]) for cand in selected_candidates
    ]

    answer_order_mode = getattr(args, "answer_evidence_order", "time")
    answer_turn_rerank_preview = []
    if answer_order_mode == "answer_aware":
        ordered_answer_candidates, answer_turn_rerank_preview = order_candidates_for_answer_aware(
            selected_candidates,
            transcripts,
            query_text,
        )
    else:
        ordered_answer_candidates = order_candidates_for_answer(
            selected_candidates,
            answer_order_mode,
        )
    ordered_answer_turns = candidate_turns_in_order(ordered_answer_candidates)

    selected_topic_rank = {
        int(topic_id): idx for idx, topic_id in enumerate(selected_topic_ids)
    }
    for cand in selected_candidates:
        matching_topic_ids = [
            int(topic_id)
            for topic_id in cand["topic_ids"]
            if int(topic_id) in selected_topic_set
        ]
        if matching_topic_ids:
            matching_topic_ids = sorted(
                matching_topic_ids,
                key=lambda tid: selected_topic_rank.get(int(tid), 10**9),
            )
            cand["transfer_topic_id"] = int(matching_topic_ids[0])
        elif cand["topic_ids"]:
            cand["transfer_topic_id"] = int(cand["topic_ids"][0])
        else:
            cand["transfer_topic_id"] = -1
        cand["transfer_node_id"] = int(
            topic_to_virtual_node_id.get(int(cand["transfer_topic_id"]), -1)
        )

    selected_virtual_node_ids = sorted(
        {
            int(cand["transfer_node_id"])
            for cand in selected_candidates
            if int(cand.get("transfer_node_id", -1)) >= 0
        }
    )
    selected_transfer_topic_ids = sorted(
        {
            int(cand.get("transfer_topic_id", -1))
            for cand in selected_candidates
            if int(cand.get("transfer_topic_id", -1)) >= 0
        }
    )
    selection_postprocess_ms = 1000.0 * (
        time.perf_counter() - selection_postprocess_start_time
    )

    route_info = {
        "granularity": "hierarchical",
        "top_level_source": "qmsum_topic_list",
        "selected_topic_strategy": "lexical",
        "hier_top_topics": args.hier_top_topics,
        "hier_topic_score_mode": args.hier_topic_score_mode,
        "hier_topic_topk": args.hier_topic_topk,
        "adaptive_topic_rescue": bool(
            getattr(args, "adaptive_topic_rescue", False)
        ),
        "adaptive_topic_rescue_triggered": bool(
            adaptive_topic_rescue_info.get("triggered", False)
        ),
        "adaptive_topic_rescue_reason": str(
            adaptive_topic_rescue_info.get("reason", "")
        ),
        "adaptive_topic_rescue_base_k": int(
            adaptive_topic_rescue_info.get("base_k", args.hier_top_topics)
        ),
        "adaptive_topic_rescue_final_k": int(
            adaptive_topic_rescue_info.get("final_k", len(selected_topic_ids))
        ),
        "adaptive_topic_rescue_gap_ratio": float(
            adaptive_topic_rescue_info.get("gap_ratio", 0.0)
        ),
        "adaptive_topic_rescue_top1_ratio": float(
            adaptive_topic_rescue_info.get("top1_ratio", 0.0)
        ),
        "adaptive_topic_rescue_top1_score": float(
            adaptive_topic_rescue_info.get("top1_score", 0.0)
        ),
        "adaptive_topic_rescue_top2_score": float(
            adaptive_topic_rescue_info.get("top2_score", 0.0)
        ),
        "adaptive_topic_rescue_margin_ratio": float(
            adaptive_topic_rescue_info.get("margin_ratio", 0.0)
        ),
        "adaptive_topic_rescue_min_top1_ratio": float(
            adaptive_topic_rescue_info.get("min_top1_ratio", 0.0)
        ),
        "topic_prototype_turns": args.topic_prototype_turns,
        "topic_representation_template": args.topic_representation_template,
        "lexical_label_repeat": args.lexical_label_repeat,
        "coarse_lexical_rerank_topk": args.coarse_lexical_rerank_topk,
        "route_chunk_size": args.route_chunk_size,
        "route_top_k": args.route_top_k,
        "effective_route_top_k": int(effective_route_top_k),
        "dynamic_route_budget": bool(getattr(args, "dynamic_route_budget", False)),
        "route_budget_mode": route_budget_mode,
        "query_budget_type": query_budget_type,
        "dynamic_candidate_pool_budget": bool(
            getattr(args, "dynamic_candidate_pool_budget", False)
        ),
        "dynamic_candidate_pool_budget_map": str(
            getattr(args, "dynamic_candidate_pool_budget_map", "")
        ),
        "dynamic_candidate_pool_min_keep": int(
            getattr(args, "dynamic_candidate_pool_min_keep", 1)
        ),
        "dynamic_candidate_pool_reason": dynamic_candidate_pool_reason,
        "dynamic_candidate_pool_target": int(dynamic_candidate_pool_target),
        "dynamic_candidate_pool_prune_ratio": float(
            dynamic_candidate_pool_prune_ratio
        ),
        "num_candidates_before_dynamic_pool": int(
            num_candidates_before_dynamic_pool
        ),
        "num_candidates_after_dynamic_pool": int(num_candidates_after_dynamic_pool),
        "num_candidates_after_coarse_segment_gate": int(
            num_candidates_after_coarse_segment_gate
        ),
        "node_summary_gate": str(
            node_summary_gate_info.get("node_summary_gate", "none")
        ),
        "node_summary_gate_summary_mode": str(
            node_summary_gate_info.get("node_summary_gate_summary_mode", "")
        ),
        "node_summary_gate_before": int(
            node_summary_gate_info.get("node_summary_gate_before", 0)
        ),
        "node_summary_gate_after": int(
            node_summary_gate_info.get("node_summary_gate_after", 0)
        ),
        "node_summary_gate_target_keep": int(
            node_summary_gate_info.get("node_summary_gate_target_keep", 0)
        ),
        "node_summary_gate_prune_ratio": float(
            node_summary_gate_info.get("node_summary_gate_prune_ratio", 0.0)
        ),
        "node_summary_gate_keep_ratio": float(
            node_summary_gate_info.get("node_summary_gate_keep_ratio", 0.0)
        ),
        "node_summary_gate_budget_mode": str(
            node_summary_gate_info.get("node_summary_gate_budget_mode", "")
        ),
        "node_summary_gate_budget_reason": str(
            node_summary_gate_info.get("node_summary_gate_budget_reason", "")
        ),
        "node_summary_gate_adaptive_min_keep": int(
            node_summary_gate_info.get("node_summary_gate_adaptive_min_keep", 0)
        ),
        "node_summary_gate_adaptive_gap": float(
            node_summary_gate_info.get("node_summary_gate_adaptive_gap", 0.0)
        ),
        "node_summary_gate_adaptive_safety_factor": float(
            node_summary_gate_info.get(
                "node_summary_gate_adaptive_safety_factor",
                0.0,
            )
        ),
        "node_summary_gate_non_finite_score_count": int(
            node_summary_gate_info.get("node_summary_gate_non_finite_score_count", 0)
        ),
        "node_summary_gate_score_pooling": str(
            node_summary_gate_info.get("node_summary_gate_score_pooling", "")
        ),
        "node_summary_gate_min_keep": int(
            node_summary_gate_info.get("node_summary_gate_min_keep", 0)
        ),
        "node_summary_gate_max_keep": int(
            node_summary_gate_info.get("node_summary_gate_max_keep", 0)
        ),
        "node_summary_gate_factor": int(
            node_summary_gate_info.get("node_summary_gate_factor", 0)
        ),
        "node_summary_gate_populate_key_cache": bool(
            node_summary_gate_info.get("node_summary_gate_populate_key_cache", False)
        ),
        "num_candidates_after_node_summary_gate": int(
            num_candidates_after_node_summary_gate
        ),
        "route_per_head": args.route_per_head,
        "route_neighbor_expand": args.route_neighbor_expand,
        "route_selection_mode": selection_mode,
        "route_hybrid_core_ratio": float(
            getattr(args, "route_hybrid_core_ratio", 0.5)
        ),
        "route_hybrid_core_max_per_turn": int(
            getattr(args, "route_hybrid_core_max_per_turn", 1)
        ),
        "route_pack_anchor_count": int(
            getattr(args, "route_pack_anchor_count", 3)
        ),
        "route_pack_support_radius": int(
            getattr(args, "route_pack_support_radius", 1)
        ),
        "route_pack_max_turns": int(
            getattr(args, "route_pack_max_turns", 10)
        ),
        "route_pack_max_candidates": int(
            getattr(args, "route_pack_max_candidates", 12)
        ),
        "route_pack_support_score_ratio": float(
            getattr(args, "route_pack_support_score_ratio", 0.55)
        ),
        "route_pack_support_same_turn": bool(
            getattr(args, "route_pack_support_same_turn", True)
        ),
        "topic_balanced_min_per_topic": int(
            getattr(args, "topic_balanced_min_per_topic", 2)
        ),
        "topic_soft_rescue_max_replacements": int(
            getattr(args, "topic_soft_rescue_max_replacements", 2)
        ),
        "topic_soft_rescue_margin_ratio": float(
            getattr(args, "topic_soft_rescue_margin_ratio", 0.15)
        ),
        "topic_soft_rescue_min_score_ratio": float(
            getattr(args, "topic_soft_rescue_min_score_ratio", 0.90)
        ),
        "route_selection_debug": route_selection_debug,
        "route_diversity_filter": args.route_diversity_filter,
        "route_diversity_max_similarity": args.route_diversity_max_similarity,
        "route_diversity_keep_ratio": args.route_diversity_keep_ratio,
        "route_diversity_min_keep": args.route_diversity_min_keep,
        "qk_aggregation": str(getattr(args, "qk_aggregation", "mean")),
        "qk_topk": int(getattr(args, "qk_topk", 4)),
        "qk_token_pooling": str(getattr(args, "qk_token_pooling", "mean")),
        "qk_query_topk_ratio": float(getattr(args, "qk_query_topk_ratio", 0.25)),
        "qk_score_batch_size": int(getattr(args, "qk_score_batch_size", 32)),
        "cache_candidate_keys": bool(getattr(args, "cache_candidate_keys", False)),
        "cache_query_q": bool(getattr(args, "cache_query_q", False)),
        "query_q_cache_enabled": bool(
            candidate_key_cache_stats.get("query_q_cache_enabled", False)
        ),
        "num_topic_nodes": len(topic_nodes),
        "num_virtual_nodes": len(virtual_node_to_topic_ids),
        "virtual_node_assignment_mode": getattr(args, "node_assignment_mode", "contiguous"),
        "topic_node_layout_path": str(getattr(args, "topic_node_layout_path", "") or ""),
        "num_candidates": len(route_candidates),
        "candidate_topic_scope": candidate_topic_scope,
        "candidate_ids_after_topic_filter": candidate_ids_after_topic_filter,
        "candidate_ids_after_prefilter": candidate_ids_after_prefilter,
        "candidate_ids_after_dynamic_pool": candidate_ids_after_dynamic_pool,
        "candidate_ids_after_coarse_segment_gate": candidate_ids_after_coarse_segment_gate,
        "candidate_ids_after_node_summary_gate": candidate_ids_after_node_summary_gate,
        "candidate_ids_scored_qk": [
            int(cand["candidate_id"]) for cand in scored_candidates
        ],
        "selected_candidate_ids": selected_candidate_ids,
        "qk_ranked_candidate_ids": qk_ranked_candidate_ids,
        "qk_ranked_candidates": [
            {
                "candidate_id": int(c["candidate_id"]),
                "turn_idx": int(c["turn_idx"]),
                "local_chunk_idx": int(c["local_chunk_idx"]),
                "score": float(c.get("score", 0.0)),
                "node_summary_score": float(
                    node_summary_scores.get(int(c["candidate_id"]), -1e9)
                ),
                "prefilter_score": float(
                    candidate_prefilter_scores.get(int(c["candidate_id"]), -1e9)
                ),
            }
            for c in qk_ranked_candidates
        ],
        "candidate_prefilter_requested_mode": candidate_prefilter_requested_mode,
        "candidate_prefilter_mode": candidate_prefilter_mode,
        "candidate_prefilter_pool_size": int(candidate_prefilter_pool_size),
        "candidate_prefilter_requested_pool_size": int(candidate_prefilter_requested_pool_size),
        "candidate_prefilter_prune_ratio": float(candidate_prefilter_prune_ratio),
        "candidate_prefilter_keep_ratio": float(
            getattr(args, "route_candidate_prefilter_keep_ratio", 0.0)
        ),
        "candidate_prefilter_min_prune_ratio": float(candidate_prefilter_min_prune_ratio),
        "candidate_prefilter_skip_reason": candidate_prefilter_skip_reason,
        "route_adaptive_prefilter": bool(
            getattr(args, "route_adaptive_prefilter", False)
        ),
        "route_adaptive_coarse_segment_gate": bool(
            getattr(args, "route_adaptive_coarse_segment_gate", False)
        ),
        "route_adaptive_min_keep_ratio": float(
            getattr(args, "route_adaptive_min_keep_ratio", 0.45)
        ),
        "route_adaptive_max_keep_ratio": float(
            getattr(args, "route_adaptive_max_keep_ratio", 0.90)
        ),
        "candidate_prefilter_adaptive_keep_ratio": float(
            candidate_prefilter_adaptive_info.get("adaptive_keep_ratio", 0.0)
        ),
        "candidate_prefilter_adaptive_uncertainty": float(
            candidate_prefilter_adaptive_info.get("adaptive_uncertainty", 0.0)
        ),
        "candidate_prefilter_adaptive_signal_count": int(
            candidate_prefilter_adaptive_info.get("adaptive_signal_count", 0)
        ),
        "candidate_prefilter_adaptive_score_span": float(
            candidate_prefilter_adaptive_info.get("adaptive_score_span", 0.0)
        ),
        "candidate_prefilter_adaptive_reason": str(
            candidate_prefilter_adaptive_info.get("adaptive_reason", "")
        ),
        "route_coarse_segment_gate": coarse_segment_gate_info.get(
            "route_coarse_segment_gate", "none"
        ),
        "coarse_segment_gate_before": int(
            coarse_segment_gate_info.get("coarse_segment_gate_before", 0)
        ),
        "coarse_segment_gate_after": int(
            coarse_segment_gate_info.get("coarse_segment_gate_after", 0)
        ),
        "coarse_segment_gate_num_segments": int(
            coarse_segment_gate_info.get("coarse_segment_gate_num_segments", 0)
        ),
        "coarse_segment_gate_keep_segments": int(
            coarse_segment_gate_info.get("coarse_segment_gate_keep_segments", 0)
        ),
        "coarse_segment_gate_segment_size": int(
            coarse_segment_gate_info.get("coarse_segment_gate_segment_size", 0)
        ),
        "coarse_segment_gate_min_keep": int(
            coarse_segment_gate_info.get("coarse_segment_gate_min_keep", 0)
        ),
        "coarse_segment_gate_ratio": float(
            coarse_segment_gate_info.get("coarse_segment_gate_ratio", 0.0)
        ),
        "coarse_segment_gate_prune_ratio": float(
            coarse_segment_gate_info.get("coarse_segment_gate_prune_ratio", 0.0)
        ),
        "coarse_segment_gate_adaptive_enabled": bool(
            coarse_segment_gate_info.get("coarse_segment_gate_adaptive_enabled", False)
        ),
        "coarse_segment_gate_adaptive_keep_ratio": float(
            coarse_segment_gate_info.get("coarse_segment_gate_adaptive_keep_ratio", 0.0)
        ),
        "coarse_segment_gate_adaptive_uncertainty": float(
            coarse_segment_gate_info.get("coarse_segment_gate_adaptive_uncertainty", 0.0)
        ),
        "coarse_segment_gate_adaptive_reason": str(
            coarse_segment_gate_info.get("coarse_segment_gate_adaptive_reason", "")
        ),
        "coarse_segment_gate_preview": coarse_segment_gate_info.get(
            "coarse_segment_gate_preview", []
        ),
        "num_candidates_before_prefilter": int(num_candidates_before_prefilter),
        "num_candidates_after_prefilter": int(num_candidates_after_prefilter),
        "num_candidates_scored_qk": len(scored_candidates),
        "num_selected_candidates": len(selected_candidates),
        "num_candidates_after_topic_filter": int(num_candidates_before_prefilter),
        "coarse_first_chunk_scope": candidate_topic_scope,
        "qk_scope": candidate_topic_scope,
        "answer_evidence_order": answer_order_mode,
        "ordered_answer_turns": ordered_answer_turns,
        "answer_turn_rerank_preview": answer_turn_rerank_preview,
        "preselected_topic_ids_before_qk": [int(x) for x in selected_topic_ids],
        "selected_topic_ids": (
            selected_transfer_topic_ids
            if candidate_topic_scope == "all_topics"
            else [int(x) for x in selected_topic_ids]
        ),
        "coarse_selected_topic_ids": [int(x) for x in selected_topic_ids],
        "selected_virtual_node_ids": selected_virtual_node_ids,
        "lexical_debug": lexical_debug,
        "selected_nodes": (
            selected_transfer_topic_ids
            if candidate_topic_scope == "all_topics"
            else [int(x) for x in selected_topic_ids]
        ),
        "selected_node_scores": {
            int(topic_id): float(topic_scores.get(topic_id, -1e9))
            for topic_id in (
                selected_transfer_topic_ids
                if candidate_topic_scope == "all_topics"
                else selected_topic_ids
            )
        },
        "virtual_node_layout": [
            {
                "node_id": int(node_id),
                "topic_ids": [int(topic_id) for topic_id in topic_ids],
            }
            for node_id, topic_ids in sorted(virtual_node_to_topic_ids.items())
        ],
        "selected_topics": [
            {
                "topic_id": int(topic["topic_id"]),
                "virtual_node_id": int(topic_to_virtual_node_id.get(int(topic["topic_id"]), -1)),
                "label": topic["label"],
                "is_gap": bool(topic["is_gap"]),
                "selection_score": float(lexical_result["scores"].get(topic["topic_id"], -1e9)),
                "qk_score": float(topic_scores.get(topic["topic_id"], -1e9)),
                "num_turns": len(topic["turns"]),
                "repr_turns": (
                    topic_index.get("topic_repr_turns", {}).get(topic["topic_id"], [])
                    if topic_index is not None
                    else []
                ),
            }
            for topic in topic_nodes
            if int(topic["topic_id"])
            in (
                set(selected_transfer_topic_ids)
                if candidate_topic_scope == "all_topics"
                else selected_topic_set
            )
        ],
        "selected_token_count": int(sum(c["n_tokens"] for c in selected_candidates)),
        "selected_candidates": [
            {
                "candidate_id": int(c["candidate_id"]),
                "turn_idx": c["turn_idx"],
                "local_chunk_idx": c["local_chunk_idx"],
                "topic_ids": c["topic_ids"],
                "transfer_topic_id": c["transfer_topic_id"],
                "transfer_node_id": int(c.get("transfer_node_id", -1)),
                "start_t": c["start_t"],
                "end_t": c["end_t"],
                "n_tokens": c["n_tokens"],
                "score": c["score"],
                "node_summary_score": float(
                    node_summary_scores.get(int(c["candidate_id"]), -1e9)
                ),
                "prefilter_score": float(
                    candidate_prefilter_scores.get(int(c["candidate_id"]), -1e9)
                ),
            }
            for c in selected_candidates[:20]
        ],
        "candidate_prefilter_preview": [
            {
                "candidate_id": int(candidate_id),
                "score": float(candidate_prefilter_scores.get(int(candidate_id), -1e9)),
            }
            for candidate_id in candidate_prefilter_selected_ids[:20]
        ],
        "dynamic_candidate_pool_preview": [
            {
                "candidate_id": int(candidate_id),
                "score": float(candidate_prefilter_scores.get(int(candidate_id), -1e9)),
            }
            for candidate_id in dynamic_candidate_pool_selected_ids[:20]
        ],
        "node_summary_gate_preview": node_summary_gate_info.get(
            "node_summary_gate_preview", []
        ),
        "routing_timing_breakdown_ms": {
            "coarse_topic_routing_ms": float(coarse_topic_routing_ms),
            "topic_filter_ms": float(topic_filter_ms),
            "candidate_prefilter_ms": float(candidate_prefilter_ms),
            "dynamic_candidate_pool_ms": float(dynamic_candidate_pool_ms),
            "coarse_segment_gate_ms": float(
                coarse_segment_gate_info.get("coarse_segment_gate_ms", 0.0)
            ),
            "node_summary_gate_ms": float(
                node_summary_gate_info.get("node_summary_gate_ms", 0.0)
            ),
            "node_summary_prepare_ms": float(
                node_summary_gate_info.get("node_summary_prepare_ms", 0.0)
            ),
            "node_summary_score_ms": float(
                node_summary_gate_info.get("node_summary_score_ms", 0.0)
            ),
            "node_summary_aggregate_ms": float(
                node_summary_gate_info.get("node_summary_aggregate_ms", 0.0)
            ),
            "candidate_key_prepare_ms": float(
                candidate_key_cache_stats.get("candidate_key_prepare_ms", 0.0)
            ),
            "query_q_prepare_ms": float(
                candidate_key_cache_stats.get("query_q_prepare_ms", 0.0)
            ),
            "qk_model_inference_ms": float(
                candidate_key_cache_stats.get("qk_model_inference_ms", 0.0)
            ),
            "qk_score_aggregation_ms": float(
                candidate_key_cache_stats.get("qk_score_aggregation_ms", 0.0)
            ),
            "qk_scoring_ms": float(qk_scoring_ms),
            "qk_total_stage_ms": float(
                candidate_key_cache_stats.get("qk_total_stage_ms", 0.0)
            ),
            "selection_postprocess_ms": float(selection_postprocess_ms),
        },
        "candidate_key_cache_stats": candidate_key_cache_stats,
    }

    strategy_results["qk"]["route_info"] = route_info
    return (
        topic_nodes,
        topic_turn_info,
        turn_to_topic_ids,
        topic_scores,
        strategy_results,
        route_info,
        selected_candidates,
    )
