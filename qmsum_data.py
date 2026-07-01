import json


def load_qmsum_sample(filepath, doc_id):
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == doc_id:
                return json.loads(line.strip())
    raise IndexError(f"doc_id={doc_id} out of range")


def _load_jsonl_sample(filepath, doc_id):
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == doc_id:
                return json.loads(line.strip())
    raise IndexError(f"doc_id={doc_id} out of range")


def _normalize_hotpotqa_sample(sample):
    context = sample.get("context") or {}
    titles = list(context.get("title") or [])
    sentence_groups = list(context.get("sentences") or [])

    transcripts = []
    title_sentence_to_turn = {}
    topic_list = []

    for title_idx, title in enumerate(titles):
        title_text = str(title).strip() or f"paragraph_{title_idx}"
        sentences = sentence_groups[title_idx] if title_idx < len(sentence_groups) else []
        start_turn = len(transcripts)

        for sent_idx, sentence in enumerate(sentences):
            sentence_text = str(sentence).strip()
            if not sentence_text:
                continue
            turn_idx = len(transcripts)
            transcripts.append(
                {
                    "speaker": title_text,
                    "content": sentence_text,
                    "source_title": title_text,
                    "source_sentence_idx": int(sent_idx),
                }
            )
            title_sentence_to_turn[(title_text, int(sent_idx))] = int(turn_idx)

        end_turn = len(transcripts) - 1
        if end_turn >= start_turn:
            topic_list.append(
                {
                    "topic": title_text,
                    "relevant_text_span": [[int(start_turn), int(end_turn)]],
                    "source_title": title_text,
                }
            )

    support = sample.get("supporting_facts") or {}
    support_titles = list(support.get("title") or [])
    support_sent_ids = list(support.get("sent_id") or [])
    relevant_turns = []
    for title, sent_id in zip(support_titles, support_sent_ids):
        title_text = str(title).strip()
        key = (title_text, int(sent_id))
        turn_idx = title_sentence_to_turn.get(key)
        if turn_idx is not None:
            relevant_turns.append(int(turn_idx))
            continue

        title_lower = title_text.lower()
        if not title_lower:
            continue
        for candidate_turn_idx, turn in enumerate(transcripts):
            content_lower = str(turn.get("content", "")).lower()
            speaker_lower = str(turn.get("speaker", "")).lower()
            if title_lower == speaker_lower or title_lower in content_lower:
                relevant_turns.append(int(candidate_turn_idx))
                break

    relevant_turns = sorted(set(relevant_turns))
    query = {
        "query": str(sample.get("question", "")).strip(),
        "answer": str(sample.get("answer", "")).strip(),
        "relevant_text_span": [[turn_idx, turn_idx] for turn_idx in relevant_turns],
        "supporting_facts": [
            {"title": str(title), "sent_id": int(sent_id)}
            for title, sent_id in zip(support_titles, support_sent_ids)
        ],
    }

    return {
        "dataset": "hotpotqa",
        "meeting_id": str(sample.get("id", "")),
        "meeting_transcripts": transcripts,
        "topic_list": topic_list,
        "specific_query_list": [query],
        "hotpotqa_type": sample.get("type", ""),
        "hotpotqa_level": sample.get("level", ""),
    }


def load_mainline_sample(filepath, doc_id, dataset="qmsum"):
    dataset_name = (dataset or "qmsum").strip().lower()
    if dataset_name == "qmsum":
        return load_qmsum_sample(filepath, doc_id)
    if dataset_name == "hotpotqa":
        return _normalize_hotpotqa_sample(_load_jsonl_sample(filepath, doc_id))
    raise ValueError(f"Unsupported dataset: {dataset}")

def _find_first_overlapping_token(offset_mapping, char_start, char_end):
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_end <= tok_start:
            continue
        if tok_end > char_start and tok_start < char_end:
            return tok_idx
    return None


def _find_last_overlapping_token(offset_mapping, char_start, char_end):
    for tok_idx in range(len(offset_mapping) - 1, -1, -1):
        tok_start, tok_end = offset_mapping[tok_idx]
        if tok_end <= tok_start:
            continue
        if tok_end > char_start and tok_start < char_end:
            return tok_idx
    return None


def build_qmsum_prompt(transcripts, tokenizer):
    parts = []
    turn_char_spans = []
    char_offset = 0
    for turn in transcripts:
        speaker = turn.get("speaker", "").strip() or "Speaker"
        content = turn.get("content", "").strip()
        turn_text = f"{speaker}: {content}\n\n"
        parts.append(turn_text)
        turn_char_spans.append((char_offset, char_offset + len(turn_text)))
        char_offset += len(turn_text)

    prompt_text = "".join(parts)
    turn_boundaries = []

    try:
        encoded = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_mapping = list(encoded["offset_mapping"])
        for char_start, char_end in turn_char_spans:
            start_t = _find_first_overlapping_token(offset_mapping, char_start, char_end)
            end_t = _find_last_overlapping_token(offset_mapping, char_start, char_end)
            if start_t is None or end_t is None:
                prev_end = turn_boundaries[-1][1] if turn_boundaries else 0
                turn_boundaries.append((prev_end, prev_end))
                continue
            turn_boundaries.append((int(start_t), int(end_t) + 1))
        return prompt_text, turn_boundaries
    except Exception:
        # Fallback for tokenizers without offset mapping support:
        # measure token spans on prefixes of the same full prompt tokenization path.
        token_offset = 0
        for _, char_end in turn_char_spans:
            prefix_ids = tokenizer(
                prompt_text[:char_end],
                add_special_tokens=False,
            ).input_ids
            prefix_len = len(prefix_ids)
            turn_boundaries.append((token_offset, prefix_len))
            token_offset = prefix_len
        return prompt_text, turn_boundaries


def format_transcript_turn(turn, turn_idx=None):
    speaker = turn.get("speaker", "").strip() or "Speaker"
    content = turn.get("content", "").strip()
    if turn_idx is None:
        return f"{speaker}: {content}"
    return f"[Turn {int(turn_idx)}] {speaker}: {content}"


def render_transcript_for_answer(transcripts, turn_indices=None):
    if turn_indices is None:
        turn_indices = range(len(transcripts))

    lines = []
    for turn_idx in turn_indices:
        if 0 <= int(turn_idx) < len(transcripts):
            lines.append(format_transcript_turn(transcripts[int(turn_idx)], turn_idx=int(turn_idx)))
    return "\n".join(lines)


def render_evidence_entries_for_answer(evidence_entries, max_entries=80, max_chars_per_entry=600):
    lines = []
    for entry in (evidence_entries or [])[: max(1, int(max_entries))]:
        turn_idx = int(entry.get("turn_idx", -1))
        speaker = entry.get("speaker", "").strip() or "Speaker"
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        max_chars = max(80, int(max_chars_per_entry))
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        lines.append(f"[Turn {turn_idx}] {speaker}: {text}")
    return "\n".join(lines)


def build_qmsum_answer_prompt(
    transcripts,
    query_text,
    turn_indices=None,
    prompt_style="basic",
):
    context_text = render_transcript_for_answer(transcripts, turn_indices=turn_indices)
    if not context_text.strip():
        context_text = "[No retrieved transcript turns.]"

    if prompt_style == "grounded":
        return (
            "Task: answer the question using only the provided context.\n"
            "Treat every context line as source evidence.\n"
            "Write a direct answer in 1-4 sentences.\n"
            "Stop after the final answer sentence.\n"
            "Do not copy or continue the Context, Evidence, Transcript, or [Turn ...] lines.\n"
            "Do not output links, web page titles, coupons, headings, lists of topics, "
            "or any text that is not an answer.\n"
            "Do not say the context is insufficient if any line contains relevant evidence.\n"
            "If there is truly no relevant evidence, write exactly: Insufficient information.\n\n"
            "Context:\n"
            f"{context_text}\n\n"
            f"Question: {query_text}\n"
            "Answer:"
        )

    if prompt_style == "strict":
        return (
            "You are answering a question about a meeting transcript.\n"
            "Use only the transcript below.\n"
            "Write one concise final answer in plain text.\n"
            "Do not repeat the question.\n"
            "Do not write new Question/Answer pairs.\n"
            "Do not invent details that are not in the transcript.\n"
            "If the transcript is insufficient, write: Insufficient information.\n\n"
            "Transcript:\n"
            f"{context_text}\n\n"
            f"Question: {query_text}\n"
            "Final answer:"
        )

    return (
        "You are given a meeting transcript.\n"
        "Answer the question using only the provided transcript.\n"
        "If the transcript does not contain enough information, say Insufficient information.\n\n"
        "Transcript:\n"
        f"{context_text}\n\n"
        f"Question: {query_text}\n"
        "Answer:"
    )


def build_qmsum_evidence_answer_prompt(
    query_text,
    evidence_entries,
    prompt_style="strict",
    max_entries=80,
    max_chars_per_entry=600,
):
    evidence_text = render_evidence_entries_for_answer(
        evidence_entries,
        max_entries=max_entries,
        max_chars_per_entry=max_chars_per_entry,
    )
    if not evidence_text.strip():
        evidence_text = "[No retrieved evidence.]"

    if prompt_style == "grounded":
        return (
            "Task: answer the question using only the retrieved evidence.\n"
            "Each evidence line is from the source context. Treat it as factual context.\n"
            "Write a direct answer in 1-4 sentences.\n"
            "Stop after the final answer sentence.\n"
            "Do not copy or continue the Context, Evidence, Transcript, or [Turn ...] lines.\n"
            "Do not output links, web page titles, coupons, headings, lists of topics, "
            "or any text that is not an answer.\n"
            "Do not say the evidence is insufficient if any line contains relevant evidence.\n"
            "If there is truly no relevant evidence, write exactly: Insufficient information.\n\n"
            "Evidence:\n"
            f"{evidence_text}\n\n"
            f"Question: {query_text}\n"
            "Answer:"
        )

    if prompt_style == "strict":
        return (
            "You are answering a question about a meeting.\n"
            "Use only the retrieved evidence below.\n"
            "Write one concise final answer in plain text.\n"
            "Do not repeat the question.\n"
            "Do not write new Question/Answer pairs.\n"
            "Do not copy irrelevant filler.\n"
            "If the evidence is insufficient, write: Insufficient information.\n\n"
            "Retrieved evidence:\n"
            f"{evidence_text}\n\n"
            f"Question: {query_text}\n"
            "Final answer:"
        )

    return (
        "You are given retrieved meeting evidence.\n"
        "Answer the question using only the evidence.\n\n"
        "Evidence:\n"
        f"{evidence_text}\n\n"
        f"Question: {query_text}\n"
        "Answer:"
    )


def spans_to_turn_set(spans):
    relevant_turns = set()
    for start_idx, end_idx in spans:
        for turn_idx in range(int(start_idx), int(end_idx) + 1):
            relevant_turns.add(int(turn_idx))
    return relevant_turns


def build_topic_nodes(meeting, num_turns):
    topic_nodes = []
    turn_to_topic_ids = [[] for _ in range(num_turns)]

    for source_topic_idx, topic_info in enumerate(meeting.get("topic_list", [])):
        turn_set = sorted(
            turn_idx
            for turn_idx in spans_to_turn_set(topic_info.get("relevant_text_span", []))
            if 0 <= turn_idx < num_turns
        )
        if not turn_set:
            continue

        topic_id = len(topic_nodes)
        topic_nodes.append(
            {
                "topic_id": int(topic_id),
                "source_topic_idx": int(source_topic_idx),
                "label": topic_info.get("topic", "").strip() or f"topic_{topic_id}",
                "turns": turn_set,
                "is_gap": False,
            }
        )
        for turn_idx in turn_set:
            turn_to_topic_ids[turn_idx].append(topic_id)

    gap_turns = []
    gap_idx = 0
    for turn_idx in range(num_turns + 1):
        uncovered = turn_idx < num_turns and not turn_to_topic_ids[turn_idx]
        if uncovered:
            gap_turns.append(turn_idx)
            continue
        if not gap_turns:
            continue

        topic_id = len(topic_nodes)
        topic_nodes.append(
            {
                "topic_id": int(topic_id),
                "source_topic_idx": -1,
                "label": f"gap_{gap_idx}",
                "turns": list(gap_turns),
                "is_gap": True,
            }
        )
        for gap_turn_idx in gap_turns:
            turn_to_topic_ids[gap_turn_idx].append(topic_id)
        gap_idx += 1
        gap_turns = []

    return topic_nodes, turn_to_topic_ids
