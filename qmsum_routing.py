"""
Legacy/general routing helpers for broad QMSum experiments.

The current focused mainline routing lives in qmsum_mainline_routing.py.
"""

from collections import defaultdict

import numpy as np

from qmsum_data import build_topic_nodes


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


def cosine_similarity(vec_a, vec_b):
    return float(
        np.dot(vec_a, vec_b)
        / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8)
    )


def aggregate_similarity_values(values, mode="mean", topk=3):
    if not values:
        return -1e9

    values = [float(v) for v in values]
    if mode == "mean":
        return float(np.mean(values))
    if mode == "topk_mean":
        k = min(max(1, int(topk)), len(values))
        return float(np.mean(sorted(values, reverse=True)[:k]))
    raise ValueError(f"Unknown similarity aggregation mode: {mode}")


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


def build_topic_lexical_document(topic, transcripts, topic_embedding_index=None, label_repeat=3):
    topic_id = int(topic["topic_id"])
    tokens = []

    if topic_embedding_index is not None:
        entry_list = topic_embedding_index.get("topic_repr_entries", {}).get(topic_id, [])
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


def build_topic_lexical_stats(
    topic_nodes,
    transcripts,
    topic_embedding_index=None,
    label_repeat=3,
):
    doc_tokens = {}
    doc_tf = {}
    doc_freq = defaultdict(int)
    doc_lens = {}

    for topic in topic_nodes:
        topic_id = int(topic["topic_id"])
        tokens = build_topic_lexical_document(
            topic,
            transcripts,
            topic_embedding_index=topic_embedding_index,
            label_repeat=label_repeat,
        )
        tf = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        doc_tokens[topic_id] = tokens
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
        "doc_tokens": doc_tokens,
        "doc_tf": doc_tf,
        "doc_lens": doc_lens,
        "avg_doc_len": avg_doc_len,
        "idf": idf,
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
    topic_embedding_index=None,
    label_repeat=3,
):
    query_tokens = tokenize_lexical_text(query_text)
    if not query_tokens:
        return {int(topic["topic_id"]): -1e9 for topic in topic_nodes}

    lexical_stats = build_topic_lexical_stats(
        topic_nodes,
        transcripts,
        topic_embedding_index=topic_embedding_index,
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


def expand_query_tokens_prf(
    query_tokens,
    topic_nodes,
    lexical_stats,
    ranked_topic_ids,
    prf_top_topics=1,
    prf_terms=4,
):
    expanded = list(query_tokens)
    existing = set(query_tokens)
    candidate_scores = defaultdict(float)
    candidate_limit = min(max(1, int(prf_top_topics)), len(ranked_topic_ids))

    topic_map = {int(topic["topic_id"]): topic for topic in topic_nodes}
    for topic_id in ranked_topic_ids[:candidate_limit]:
        topic = topic_map.get(int(topic_id))
        if topic is None or topic.get("is_gap", False):
            continue
        tf_map = lexical_stats["doc_tf"].get(int(topic_id), {})
        for tok, tf in tf_map.items():
            if tok in existing:
                continue
            candidate_scores[tok] += float(tf) * float(lexical_stats["idf"].get(tok, 0.0))

    ranked_terms = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))
    for tok, _ in ranked_terms[: max(1, int(prf_terms))]:
        expanded.append(tok)
    return expanded


def score_topics_lexical_prf_qmsum(
    query_text,
    topic_nodes,
    transcripts,
    topic_embedding_index=None,
    label_repeat=3,
    prf_top_topics=1,
    prf_terms=4,
):
    query_tokens = tokenize_lexical_text(query_text)
    if not query_tokens:
        return {int(topic["topic_id"]): -1e9 for topic in topic_nodes}

    lexical_stats = build_topic_lexical_stats(
        topic_nodes,
        transcripts,
        topic_embedding_index=topic_embedding_index,
        label_repeat=label_repeat,
    )
    initial_scores = {}
    for topic in topic_nodes:
        topic_id = int(topic["topic_id"])
        initial_scores[topic_id] = bm25_score_tokens(
            query_tokens,
            lexical_stats["doc_tf"].get(topic_id, {}),
            lexical_stats["doc_lens"].get(topic_id, 0),
            lexical_stats["avg_doc_len"],
            lexical_stats["idf"],
        )

    ranked_topic_ids = [
        topic_id
        for topic_id, score in sorted(
            initial_scores.items(),
            key=lambda x: (-float(x[1]), int(x[0])),
        )
        if float(score) > -1e8
    ]
    expanded_query_tokens = expand_query_tokens_prf(
        query_tokens,
        topic_nodes,
        lexical_stats,
        ranked_topic_ids,
        prf_top_topics=prf_top_topics,
        prf_terms=prf_terms,
    )

    final_scores = {}
    for topic in topic_nodes:
        topic_id = int(topic["topic_id"])
        score = bm25_score_tokens(
            expanded_query_tokens,
            lexical_stats["doc_tf"].get(topic_id, {}),
            lexical_stats["doc_lens"].get(topic_id, 0),
            lexical_stats["avg_doc_len"],
            lexical_stats["idf"],
        )
        if topic.get("is_gap", False) and score > -1e8:
            score -= 0.05
        final_scores[topic_id] = float(score)
    return final_scores


def score_nodes_embedding_qmsum(query_text, node_turn_info, transcripts):
    from distributed_sim import _get_embedding_model

    model = _get_embedding_model()
    if model is None:
        return {n: -1e9 for n in node_turn_info}

    query_emb = model.encode([query_text])[0]
    turn_texts = [
        f"{turn.get('speaker', '').strip()}: {turn.get('content', '').strip()}".strip()
        for turn in transcripts
    ]
    turn_embeddings = model.encode(turn_texts) if turn_texts else []

    scores = {}
    for node_id, turns in node_turn_info.items():
        if not turns:
            scores[node_id] = -1e9
            continue

        sims = []
        for turn_idx in turns:
            if turn_idx >= len(turn_texts):
                continue
            text = turn_texts[turn_idx]
            if not text:
                continue
            text_emb = turn_embeddings[turn_idx]
            sim = cosine_similarity(query_emb, text_emb)
            sims.append(sim)
        scores[node_id] = float(np.mean(sims)) if sims else -1e9
    return scores


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


def build_topic_speaker_summary(topic, transcripts, max_speakers=3):
    speaker_counts = defaultdict(int)
    for turn_idx in topic.get("turns", []):
        if 0 <= int(turn_idx) < len(transcripts):
            speaker = transcripts[int(turn_idx)].get("speaker", "").strip() or "Speaker"
            speaker_counts[speaker] += 1

    ranked = sorted(speaker_counts.items(), key=lambda x: (-x[1], x[0]))
    return ranked[: max(1, int(max_speakers))]


def build_topic_representation_text(
    topic,
    transcripts,
    prototype_turns,
    representation_template="basic",
):
    lines = []
    label = topic.get("label", "").strip()
    if label:
        lines.append(f"Topic: {label}")
    if topic.get("is_gap", False):
        lines.append("Type: gap / unlabeled conversation region")

    sampled_turns = sample_topic_turn_indices(topic.get("turns", []), prototype_turns)
    if representation_template == "enhanced":
        turns = topic.get("turns", [])
        if turns:
            lines.append(
                f"Topic span: turns {int(turns[0])} to {int(turns[-1])} "
                f"({len(turns)} turns)"
            )

        top_speakers = build_topic_speaker_summary(topic, transcripts, max_speakers=3)
        if top_speakers:
            speaker_text = ", ".join(
                f"{speaker} ({count})" for speaker, count in top_speakers
            )
            lines.append(f"Main speakers: {speaker_text}")

        if sampled_turns:
            lines.append("Representative turns across this topic region:")
        for idx, turn_idx in enumerate(sampled_turns):
            if 0 <= int(turn_idx) < len(transcripts):
                turn = transcripts[int(turn_idx)]
                speaker = turn.get("speaker", "").strip() or "Speaker"
                content = turn.get("content", "").strip()
                slot = ["early", "middle", "late"]
                slot_name = slot[min(idx, len(slot) - 1)] if len(sampled_turns) <= 3 else f"sample{idx+1}"
                lines.append(
                    f"[{slot_name} turn {int(turn_idx)}] {speaker}: {content}"
                )
    else:
        if sampled_turns:
            lines.append("Representative turns:")
        for turn_idx in sampled_turns:
            if 0 <= int(turn_idx) < len(transcripts):
                turn = transcripts[int(turn_idx)]
                speaker = turn.get("speaker", "").strip() or "Speaker"
                content = turn.get("content", "").strip()
                lines.append(f"[Turn {int(turn_idx)}] {speaker}: {content}")

    return "\n".join(lines).strip(), sampled_turns


def build_topic_embedding_index_qmsum(
    meeting,
    transcripts,
    topic_embedding_source="precomputed_topic_text",
    topic_prototype_turns=3,
    topic_representation_template="basic",
):
    topic_nodes, turn_to_topic_ids = build_topic_nodes(meeting, num_turns=len(transcripts))
    topic_turn_info = {topic["topic_id"]: topic["turns"] for topic in topic_nodes}

    topic_embedding_index = {
        "source": topic_embedding_source,
        "topic_nodes": topic_nodes,
        "turn_to_topic_ids": turn_to_topic_ids,
        "topic_turn_info": topic_turn_info,
        "topic_repr_texts": {},
        "topic_repr_turns": {},
        "topic_repr_entries": {},
    }

    if topic_embedding_source == "online_turns":
        return topic_embedding_index

    from distributed_sim import _get_embedding_model

    model = _get_embedding_model()
    if model is None:
        return topic_embedding_index

    repr_texts = []
    repr_keys = []
    for topic in topic_nodes:
        topic_id = int(topic["topic_id"])
        repr_text, sampled_turns = build_topic_representation_text(
            topic,
            transcripts,
            prototype_turns=max(1, int(topic_prototype_turns)),
            representation_template=topic_representation_template,
        )
        topic_embedding_index["topic_repr_texts"][topic_id] = repr_text
        topic_embedding_index["topic_repr_turns"][topic_id] = sampled_turns
        entry_list = []

        label = topic.get("label", "").strip()
        if label:
            entry_list.append(
                {
                    "entry_type": "label",
                    "text": f"Topic label: {label}",
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

        topic_embedding_index["topic_repr_entries"][topic_id] = entry_list
        for entry_idx, entry in enumerate(entry_list):
            if entry.get("text"):
                repr_texts.append(entry["text"])
                repr_keys.append((topic_id, entry_idx))

    if repr_texts:
        repr_embeddings = model.encode(repr_texts)
        for (topic_id, entry_idx), emb in zip(repr_keys, repr_embeddings):
            topic_embedding_index["topic_repr_entries"][int(topic_id)][int(entry_idx)]["embedding"] = emb

    return topic_embedding_index


def score_topics_embedding_qmsum(
    query_text,
    topic_nodes,
    transcripts,
    turn_score_mode="topk_mean",
    turn_topk=3,
    label_weight=0.35,
    topic_embedding_index=None,
):
    from distributed_sim import _get_embedding_model

    model = _get_embedding_model()
    if model is None:
        return {topic["topic_id"]: -1e9 for topic in topic_nodes}

    query_emb = model.encode([query_text])[0]
    if (
        topic_embedding_index is not None
        and topic_embedding_index.get("source") == "precomputed_topic_text"
    ):
        scores = {}
        for topic in topic_nodes:
            topic_id = int(topic["topic_id"])
            entry_list = topic_embedding_index.get("topic_repr_entries", {}).get(topic_id, [])
            sim_values = []
            for entry in entry_list:
                entry_emb = entry.get("embedding")
                if entry_emb is None:
                    continue
                sim = cosine_similarity(query_emb, entry_emb)
                if entry.get("entry_type") == "label":
                    sim += 0.03
                sim_values.append(float(sim))
            if not sim_values:
                scores[topic_id] = -1e9
                continue
            score = aggregate_similarity_values(
                sim_values,
                mode=turn_score_mode,
                topk=turn_topk,
            )
            if topic.get("is_gap", False):
                score -= 0.05
            scores[topic_id] = float(score)
        return scores

    turn_texts = [
        f"{turn.get('speaker', '').strip()}: {turn.get('content', '').strip()}".strip()
        for turn in transcripts
    ]
    turn_embeddings = model.encode(turn_texts) if turn_texts else []

    label_inputs = []
    label_topic_ids = []
    for topic in topic_nodes:
        label = topic.get("label", "").strip()
        if label and not topic.get("is_gap", False):
            label_inputs.append(label)
            label_topic_ids.append(int(topic["topic_id"]))
    label_score_map = {}
    if label_inputs:
        label_embeddings = model.encode(label_inputs)
        for topic_id, label_emb in zip(label_topic_ids, label_embeddings):
            label_score_map[topic_id] = cosine_similarity(query_emb, label_emb)

    scores = {}
    for topic in topic_nodes:
        topic_id = int(topic["topic_id"])
        turns = topic.get("turns", [])
        if not turns:
            scores[topic_id] = -1e9
            continue

        turn_sims = []
        for turn_idx in turns:
            if 0 <= turn_idx < len(turn_embeddings):
                turn_sims.append(cosine_similarity(query_emb, turn_embeddings[turn_idx]))

        if not turn_sims:
            scores[topic_id] = -1e9
            continue

        turn_score = aggregate_similarity_values(
            turn_sims,
            mode=turn_score_mode,
            topk=turn_topk,
        )
        if topic.get("is_gap", False):
            scores[topic_id] = float(turn_score - 0.05)
            continue

        topic_label_weight = min(max(float(label_weight), 0.0), 1.0)
        label_score = label_score_map.get(topic_id)
        if label_score is None:
            scores[topic_id] = float(turn_score)
        else:
            scores[topic_id] = float(
                (1.0 - topic_label_weight) * turn_score
                + topic_label_weight * float(label_score)
            )
    return scores


def normalize_score_dict(score_dict, valid_floor=-1e8):
    valid_items = [(k, float(v)) for k, v in score_dict.items() if float(v) > valid_floor]
    if not valid_items:
        return {k: 0.0 for k in score_dict}

    values = np.asarray([v for _, v in valid_items], dtype=np.float32)
    v_min = float(np.min(values))
    v_max = float(np.max(values))

    if v_max - v_min < 1e-8:
        return {
            k: (0.0 if float(v) <= valid_floor else 1.0)
            for k, v in score_dict.items()
        }

    normalized = {}
    for key, value in score_dict.items():
        value = float(value)
        if value <= valid_floor:
            normalized[key] = 0.0
        else:
            normalized[key] = float((value - v_min) / (v_max - v_min))
    return normalized


def build_rrf_topic_scores(score_dicts, rank_constant=60):
    rankings = []
    for score_dict in score_dicts:
        ranked = [
            topic_id
            for topic_id, score in sorted(
                score_dict.items(),
                key=lambda x: (-float(x[1]), int(x[0])),
            )
            if float(score) > -1e8
        ]
        rankings.append(ranked)

    topic_ids = set()
    for score_dict in score_dicts:
        topic_ids.update(int(k) for k in score_dict.keys())

    rrf_scores = {}
    for topic_id in topic_ids:
        score = 0.0
        for ranked in rankings:
            if int(topic_id) in ranked:
                rank = ranked.index(int(topic_id)) + 1
                score += 1.0 / (float(rank_constant) + float(rank))
        rrf_scores[int(topic_id)] = float(score) if score > 0.0 else -1e9
    return rrf_scores


def compute_candidate_turn_rerank_scores(
    query_text,
    topic_nodes,
    transcripts,
    candidate_topic_ids,
    turn_score_mode="topk_mean",
    turn_topk=3,
):
    from distributed_sim import _get_embedding_model

    model = _get_embedding_model()
    if model is None:
        return {int(topic["topic_id"]): -1e9 for topic in topic_nodes}

    candidate_set = {int(topic_id) for topic_id in candidate_topic_ids}
    query_emb = model.encode([query_text])[0]

    needed_turn_ids = sorted(
        {
            int(turn_idx)
            for topic in topic_nodes
            if int(topic["topic_id"]) in candidate_set
            for turn_idx in topic.get("turns", [])
            if 0 <= int(turn_idx) < len(transcripts)
        }
    )
    if not needed_turn_ids:
        return {int(topic["topic_id"]): -1e9 for topic in topic_nodes}

    turn_texts = []
    for turn_idx in needed_turn_ids:
        turn = transcripts[int(turn_idx)]
        speaker = turn.get("speaker", "").strip() or "Speaker"
        content = turn.get("content", "").strip()
        turn_texts.append(f"{speaker}: {content}")

    turn_embeddings = model.encode(turn_texts)
    turn_emb_map = {
        int(turn_idx): emb for turn_idx, emb in zip(needed_turn_ids, turn_embeddings)
    }

    rerank_scores = {}
    for topic in topic_nodes:
        topic_id = int(topic["topic_id"])
        if topic_id not in candidate_set:
            rerank_scores[topic_id] = -1e9
            continue

        sim_values = []
        for turn_idx in topic.get("turns", []):
            emb = turn_emb_map.get(int(turn_idx))
            if emb is None:
                continue
            sim_values.append(cosine_similarity(query_emb, emb))

        if not sim_values:
            rerank_scores[topic_id] = -1e9
            continue

        rerank_scores[topic_id] = aggregate_similarity_values(
            sim_values,
            mode=turn_score_mode,
            topk=turn_topk,
        )

    return rerank_scores


def build_rerank_topic_scores(
    embedding_scores,
    qk_scores,
    candidate_topic_ids,
    embedding_weight,
    qk_weight,
):
    emb_norm = normalize_score_dict(embedding_scores)
    qk_norm = normalize_score_dict(qk_scores)
    candidate_set = {int(topic_id) for topic_id in candidate_topic_ids}

    rerank_scores = {}
    for topic_id in embedding_scores.keys():
        if int(topic_id) not in candidate_set:
            rerank_scores[topic_id] = -1e9
            continue
        rerank_scores[topic_id] = float(
            embedding_weight * emb_norm.get(topic_id, 0.0)
            + qk_weight * qk_norm.get(topic_id, 0.0)
        )

    return rerank_scores, emb_norm, qk_norm


def score_hierarchical_topic_chunk(
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
    topic_embedding_index=None,
):
    from distributed_sim import (
        aggregate_qk_scores,
        rank_scores,
        score_nodes_random,
        score_nodes_recency,
        stack_keys_from_kv,
    )

    if topic_embedding_index is not None:
        topic_nodes = topic_embedding_index["topic_nodes"]
        turn_to_topic_ids = topic_embedding_index["turn_to_topic_ids"]
        topic_turn_info = topic_embedding_index["topic_turn_info"]
    else:
        topic_nodes, turn_to_topic_ids = build_topic_nodes(meeting, num_turns=len(transcripts))
        topic_turn_info = {topic["topic_id"]: topic["turns"] for topic in topic_nodes}
    route_candidates = build_hierarchical_candidates(
        turn_boundaries,
        total_tokens,
        turn_to_topic_ids,
        max(1, args.route_chunk_size),
    )

    need_random = args.baselines or args.hier_top_strategy == "random"
    need_recency = args.baselines or args.hier_top_strategy == "recency"
    need_embedding = args.baselines or args.hier_top_strategy in [
        "embedding",
        "rerank",
        "lexical_hybrid",
        "rrf",
    ]
    need_lexical = args.baselines or args.hier_top_strategy in [
        "lexical",
        "lexical_prf",
        "lexical_hybrid",
        "rrf",
    ]

    topic_strategy_scores = {}
    if need_random:
        topic_strategy_scores["random"] = score_nodes_random(
            len(topic_nodes), seed=args._current_seed
        )
    if need_recency:
        topic_strategy_scores["recency"] = score_nodes_recency(
            topic_turn_info, len(topic_nodes)
        )
    if need_embedding:
        topic_strategy_scores["embedding"] = score_topics_embedding_qmsum(
            query_text,
            topic_nodes,
            transcripts,
            turn_score_mode=args.topic_embedding_turn_score_mode,
            turn_topk=args.topic_embedding_topk,
            label_weight=args.topic_label_weight,
            topic_embedding_index=topic_embedding_index,
        )
    if need_lexical:
        topic_strategy_scores["lexical"] = score_topics_lexical_qmsum(
            query_text,
            topic_nodes,
            transcripts,
            topic_embedding_index=topic_embedding_index,
            label_repeat=args.lexical_label_repeat,
        )
        topic_strategy_scores["lexical_prf"] = score_topics_lexical_prf_qmsum(
            query_text,
            topic_nodes,
            transcripts,
            topic_embedding_index=topic_embedding_index,
            label_repeat=args.lexical_label_repeat,
            prf_top_topics=args.lexical_prf_top_topics,
            prf_terms=args.lexical_prf_terms,
        )
        if "embedding" in topic_strategy_scores:
            topic_strategy_scores["lexical_hybrid"] = build_rerank_topic_scores(
                topic_strategy_scores["embedding"],
                topic_strategy_scores["lexical"],
                candidate_topic_ids=[topic["topic_id"] for topic in topic_nodes],
                embedding_weight=args.lexical_hybrid_embedding_weight,
                qk_weight=args.lexical_hybrid_lexical_weight,
            )[0]
            topic_strategy_scores["rrf"] = build_rrf_topic_scores(
                [
                    topic_strategy_scores["embedding"],
                    topic_strategy_scores["lexical"],
                    topic_strategy_scores["lexical_prf"],
                ],
                rank_constant=args.rrf_k,
            )

    preselected_topic_ids = []
    preselected_topic_set = None
    can_preselect_before_chunk_qk = (
        not args.baselines
        and args.hier_top_strategy in topic_strategy_scores
        and args.hier_top_strategy not in {"qk", "rerank"}
    )
    if can_preselect_before_chunk_qk:
        k_topics = min(max(1, args.hier_top_topics), len(topic_nodes))
        preselected_topic_ids = rank_scores(
            topic_strategy_scores[args.hier_top_strategy]
        )["ranked_nodes"][:k_topics]
        preselected_topic_set = set(preselected_topic_ids)

    topic_scores = {topic["topic_id"]: -1e9 for topic in topic_nodes}
    scored_candidates = []
    per_head_scores = []
    topic_score_buckets = defaultdict(list)

    candidates_to_score = route_candidates
    if preselected_topic_set is not None:
        candidates_to_score = [
            cand
            for cand in route_candidates
            if any(int(topic_id) in preselected_topic_set for topic_id in cand["topic_ids"])
        ]

    for cand in candidates_to_score:
        kv_chunk = split_kv(kv_3d, cand["start_t"], cand["end_t"])
        k_stacked = stack_keys_from_kv(kv_chunk)
        scores, _ = qk_score_fn(query_ids, [k_stacked], model, scoring_layers)
        scalar = aggregate_qk_scores(scores, args.qk_aggregation, args.qk_topk)

        record = dict(cand)
        record["score"] = float(scalar)
        scored_candidates.append(record)
        per_head_scores.append(np.asarray(scores, dtype=np.float32).reshape(-1))

        for topic_id in record["topic_ids"]:
            topic_score_buckets[topic_id].append(float(scalar))

        del kv_chunk, k_stacked, scores

    for topic_id, bucket in topic_score_buckets.items():
        if args.hier_topic_score_mode == "sum":
            topic_scores[topic_id] = float(sum(bucket))
        elif args.hier_topic_score_mode == "max":
            topic_scores[topic_id] = float(max(bucket))
        elif args.hier_topic_score_mode == "topk_mean":
            topic_scores[topic_id] = float(
                aggregate_qk_scores(bucket, mode="topk_mean", topk=args.hier_topic_topk)
            )
        else:
            raise ValueError(f"Unknown hier_topic_score_mode: {args.hier_topic_score_mode}")

    topic_strategy_scores["qk"] = topic_scores
    if args.hier_top_strategy == "rerank":
        if "embedding" not in topic_strategy_scores:
            topic_strategy_scores["embedding"] = score_topics_embedding_qmsum(
                query_text,
                topic_nodes,
                transcripts,
                turn_score_mode=args.topic_embedding_turn_score_mode,
                turn_topk=args.topic_embedding_topk,
                label_weight=args.topic_label_weight,
                topic_embedding_index=topic_embedding_index,
            )
        embedding_ranked = rank_scores(topic_strategy_scores["embedding"])["ranked_nodes"]
        rerank_candidate_count = min(
            max(args.hier_top_topics, args.rerank_candidate_topics),
            len(topic_nodes),
        )
        rerank_candidate_topic_ids = embedding_ranked[:rerank_candidate_count]
        if (
            args.topic_embedding_source == "precomputed_topic_text"
            and args.rerank_source == "candidate_turns"
        ):
            candidate_turn_scores = compute_candidate_turn_rerank_scores(
                query_text,
                topic_nodes,
                transcripts,
                rerank_candidate_topic_ids,
                turn_score_mode=args.topic_embedding_turn_score_mode,
                turn_topk=args.topic_embedding_topk,
            )
            rerank_scores, emb_norm, qk_norm = build_rerank_topic_scores(
                topic_strategy_scores["embedding"],
                candidate_turn_scores,
                rerank_candidate_topic_ids,
                embedding_weight=args.rerank_embedding_weight,
                qk_weight=args.rerank_qk_weight,
            )
            topic_strategy_scores["candidate_turn_rerank"] = candidate_turn_scores
        else:
            rerank_scores, emb_norm, qk_norm = build_rerank_topic_scores(
                topic_strategy_scores["embedding"],
                topic_scores,
                rerank_candidate_topic_ids,
                embedding_weight=args.rerank_embedding_weight,
                qk_weight=args.rerank_qk_weight,
            )
        topic_strategy_scores["rerank"] = rerank_scores
    else:
        emb_norm = None
        qk_norm = None
        rerank_candidate_topic_ids = []

    strategy_results = {
        strat_name: rank_scores(scores)
        for strat_name, scores in topic_strategy_scores.items()
    }

    k_topics = min(max(1, args.hier_top_topics), len(topic_nodes))
    if args.hier_top_strategy not in strategy_results:
        raise ValueError(
            f"Top-level strategy {args.hier_top_strategy!r} is unavailable. "
            "Enable baselines or choose qk."
        )
    selected_topic_ids = strategy_results[args.hier_top_strategy]["ranked_nodes"][:k_topics]
    selected_topic_set = set(selected_topic_ids)
    selected_topic_rank = {
        int(topic_id): idx for idx, topic_id in enumerate(selected_topic_ids)
    }

    filtered_indices = [
        idx
        for idx, cand in enumerate(scored_candidates)
        if any(topic_id in selected_topic_set for topic_id in cand["topic_ids"])
    ]

    selected_candidates = []
    if filtered_indices:
        if args.route_per_head and per_head_scores:
            score_matrix = np.stack([per_head_scores[idx] for idx in filtered_indices], axis=0)
            selected_local_ids = set()
            k = min(max(1, args.route_top_k), len(filtered_indices))
            for head_idx in range(score_matrix.shape[1]):
                top_local = np.argsort(score_matrix[:, head_idx])[-k:]
                selected_local_ids.update(int(i) for i in top_local)
            selected_candidates = [
                scored_candidates[filtered_indices[i]] for i in selected_local_ids
            ]
            selected_candidates = sorted(selected_candidates, key=lambda x: -x["score"])
        else:
            k = min(max(1, args.route_top_k), len(filtered_indices))
            selected_candidates = sorted(
                (scored_candidates[idx] for idx in filtered_indices),
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
        selected_topic_only = {int(tid) for tid in selected_topic_set}

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
                    if not any(int(tid) in selected_topic_only for tid in neighbor["topic_ids"]):
                        continue
                    expanded_candidates.append(neighbor)
                    expanded_keys.add(key)

        selected_candidates = sorted(expanded_candidates, key=lambda x: -x["score"])

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

    route_info = {
        "granularity": "hierarchical",
        "top_level_source": "qmsum_topic_list",
        "selected_topic_strategy": args.hier_top_strategy,
        "hier_top_topics": args.hier_top_topics,
        "hier_topic_score_mode": args.hier_topic_score_mode,
        "hier_topic_topk": args.hier_topic_topk,
        "topic_embedding_turn_score_mode": args.topic_embedding_turn_score_mode,
        "topic_embedding_topk": args.topic_embedding_topk,
        "topic_label_weight": args.topic_label_weight,
        "topic_embedding_source": args.topic_embedding_source,
        "topic_prototype_turns": args.topic_prototype_turns,
        "lexical_label_repeat": args.lexical_label_repeat,
        "lexical_prf_top_topics": args.lexical_prf_top_topics,
        "lexical_prf_terms": args.lexical_prf_terms,
        "lexical_hybrid_embedding_weight": args.lexical_hybrid_embedding_weight,
        "lexical_hybrid_lexical_weight": args.lexical_hybrid_lexical_weight,
        "rrf_k": args.rrf_k,
        "rerank_candidate_topics": args.rerank_candidate_topics,
        "rerank_source": args.rerank_source,
        "rerank_embedding_weight": args.rerank_embedding_weight,
        "rerank_qk_weight": args.rerank_qk_weight,
        "route_chunk_size": args.route_chunk_size,
        "route_top_k": args.route_top_k,
        "route_per_head": args.route_per_head,
        "route_neighbor_expand": args.route_neighbor_expand,
        "num_topic_nodes": len(topic_nodes),
        "num_candidates": len(route_candidates),
        "num_candidates_scored_qk": len(scored_candidates),
        "num_selected_candidates": len(selected_candidates),
        "num_candidates_after_topic_filter": len(filtered_indices),
        "coarse_first_chunk_scope": (
            "selected_topics_only" if preselected_topic_set is not None else "all_topics"
        ),
        "preselected_topic_ids_before_qk": [int(x) for x in preselected_topic_ids],
        "selected_topic_ids": [int(x) for x in selected_topic_ids],
        "selected_nodes": [int(x) for x in selected_topic_ids],
        "rerank_candidate_topic_ids": [
            int(x) for x in (rerank_candidate_topic_ids if args.hier_top_strategy == "rerank" else [])
        ],
        "selected_node_scores": {
            int(topic_id): float(topic_scores.get(topic_id, -1e9))
            for topic_id in selected_topic_ids
        },
        "selected_topics": [
            {
                "topic_id": int(topic["topic_id"]),
                "label": topic["label"],
                "is_gap": bool(topic["is_gap"]),
                "selection_score": float(
                    topic_strategy_scores[args.hier_top_strategy].get(topic["topic_id"], -1e9)
                ),
                "qk_score": float(topic_scores.get(topic["topic_id"], -1e9)),
                "candidate_turn_rerank_score": float(
                    topic_strategy_scores.get("candidate_turn_rerank", {}).get(topic["topic_id"], -1e9)
                ),
                "embedding_score": float(
                    topic_strategy_scores.get("embedding", {}).get(topic["topic_id"], -1e9)
                ),
                "embedding_score_norm": (
                    float(emb_norm.get(topic["topic_id"], 0.0)) if emb_norm is not None else None
                ),
                "qk_score_norm": (
                    float(qk_norm.get(topic["topic_id"], 0.0)) if qk_norm is not None else None
                ),
                "is_rerank_candidate": bool(
                    topic["topic_id"] in (
                        rerank_candidate_topic_ids if args.hier_top_strategy == "rerank" else []
                    )
                ),
                "num_turns": len(topic["turns"]),
                "repr_turns": (
                    topic_embedding_index.get("topic_repr_turns", {}).get(topic["topic_id"], [])
                    if topic_embedding_index is not None
                    else []
                ),
            }
            for topic in topic_nodes
            if topic["topic_id"] in selected_topic_set
        ],
        "selected_token_count": int(sum(c["n_tokens"] for c in selected_candidates)),
        "selected_candidates": [
            {
                "turn_idx": c["turn_idx"],
                "local_chunk_idx": c["local_chunk_idx"],
                "topic_ids": c["topic_ids"],
                "transfer_topic_id": c["transfer_topic_id"],
                "start_t": c["start_t"],
                "end_t": c["end_t"],
                "n_tokens": c["n_tokens"],
                "score": c["score"],
            }
            for c in selected_candidates[:20]
        ],
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
