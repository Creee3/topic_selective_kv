import json
import os
import time
from collections import defaultdict

import numpy as np

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
        }

    start_time = time.perf_counter()
    segment_size = max(1, int(getattr(args, "route_coarse_segment_size", 4)))
    keep_ratio = min(
        1.0,
        max(0.0, float(getattr(args, "route_coarse_segment_keep_ratio", 0.5))),
    )
    min_keep = max(1, int(getattr(args, "route_coarse_segment_min_keep", 32)))
    max_keep = int(getattr(args, "route_coarse_segment_max_keep", 0))

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
        }

    target_keep_candidates = max(min_keep, int(np.ceil(len(candidates) * keep_ratio)))
    if max_keep > 0:
        target_keep_candidates = min(target_keep_candidates, max_keep)
    target_keep_candidates = min(len(candidates), max(1, target_keep_candidates))

    if target_keep_candidates >= len(candidates):
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
    }


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
            )
        else:
            scores_matrix, _ = qk_score_fn(
                query_ids,
                chunk_keys_list,
                model,
                scoring_layers,
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

    k_topics = min(max(1, args.hier_top_topics), len(topic_nodes))
    selected_topic_ids = lexical_ranked_topics[:k_topics]
    selected_topic_set = set(int(x) for x in selected_topic_ids)
    coarse_topic_routing_ms = 1000.0 * (
        time.perf_counter() - coarse_topic_routing_start_time
    )

    topic_filter_start_time = time.perf_counter()
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

    if candidate_prefilter_requested_mode == "lexical" and candidates_to_score:
        candidate_prefilter_requested_pool_size = resolve_candidate_prefilter_size(
            args,
            effective_route_top_k,
            len(candidates_to_score),
        )
        candidate_prefilter_prune_ratio = 1.0 - (
            float(candidate_prefilter_requested_pool_size)
            / max(1.0, float(num_candidates_before_prefilter))
        )
        if candidate_prefilter_prune_ratio < candidate_prefilter_min_prune_ratio:
            candidate_prefilter_mode = "lexical_skipped_low_prune"
            candidate_prefilter_pool_size = len(candidates_to_score)
            candidate_prefilter_skip_reason = (
                "requested pool would prune too few candidates"
            )
            for cand in candidates_to_score:
                cand["prefilter_score"] = -1e9
        else:
            candidate_prefilter_pool_size = candidate_prefilter_requested_pool_size
            candidate_ids = [int(cand["candidate_id"]) for cand in candidates_to_score]
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
    if bool(getattr(args, "cache_query_q", False)) and candidates_to_score:
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
            if topic_id in selected_topic_set:
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

    selected_candidates = []
    if scored_candidates:
        if args.route_per_head and per_head_scores:
            score_matrix = np.stack(per_head_scores, axis=0)
            selected_local_ids = set()
            k = min(effective_route_top_k, len(scored_candidates))
            for head_idx in range(score_matrix.shape[1]):
                top_local = np.argsort(score_matrix[:, head_idx])[-k:]
                selected_local_ids.update(int(i) for i in top_local)
            # Keep the final routing budget hard-capped even under per-head voting.
            selected_candidates = [scored_candidates[i] for i in selected_local_ids]
            selected_candidates = sorted(selected_candidates, key=lambda x: -x["score"])[:k]
        else:
            k = min(effective_route_top_k, len(scored_candidates))
            selected_candidates = sorted(
                scored_candidates,
                key=lambda x: -x["score"],
            )[:k]

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
        "route_per_head": args.route_per_head,
        "route_neighbor_expand": args.route_neighbor_expand,
        "route_diversity_filter": args.route_diversity_filter,
        "route_diversity_max_similarity": args.route_diversity_max_similarity,
        "route_diversity_keep_ratio": args.route_diversity_keep_ratio,
        "route_diversity_min_keep": args.route_diversity_min_keep,
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
        "candidate_ids_after_topic_filter": candidate_ids_after_topic_filter,
        "candidate_ids_after_prefilter": candidate_ids_after_prefilter,
        "candidate_ids_after_dynamic_pool": candidate_ids_after_dynamic_pool,
        "candidate_ids_after_coarse_segment_gate": candidate_ids_after_coarse_segment_gate,
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
        "coarse_segment_gate_preview": coarse_segment_gate_info.get(
            "coarse_segment_gate_preview", []
        ),
        "num_candidates_before_prefilter": int(num_candidates_before_prefilter),
        "num_candidates_after_prefilter": int(num_candidates_after_prefilter),
        "num_candidates_scored_qk": len(scored_candidates),
        "num_selected_candidates": len(selected_candidates),
        "num_candidates_after_topic_filter": int(num_candidates_before_prefilter),
        "coarse_first_chunk_scope": "selected_topics_only",
        "qk_scope": "selected_topics_only",
        "answer_evidence_order": answer_order_mode,
        "ordered_answer_turns": ordered_answer_turns,
        "answer_turn_rerank_preview": answer_turn_rerank_preview,
        "preselected_topic_ids_before_qk": [int(x) for x in selected_topic_ids],
        "selected_topic_ids": [int(x) for x in selected_topic_ids],
        "selected_virtual_node_ids": selected_virtual_node_ids,
        "lexical_debug": lexical_debug,
        "selected_nodes": [int(x) for x in selected_topic_ids],
        "selected_node_scores": {
            int(topic_id): float(topic_scores.get(topic_id, -1e9))
            for topic_id in selected_topic_ids
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
            if int(topic["topic_id"]) in selected_topic_set
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
        "routing_timing_breakdown_ms": {
            "coarse_topic_routing_ms": float(coarse_topic_routing_ms),
            "topic_filter_ms": float(topic_filter_ms),
            "candidate_prefilter_ms": float(candidate_prefilter_ms),
            "dynamic_candidate_pool_ms": float(dynamic_candidate_pool_ms),
            "coarse_segment_gate_ms": float(
                coarse_segment_gate_info.get("coarse_segment_gate_ms", 0.0)
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
