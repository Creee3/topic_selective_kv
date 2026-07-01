from collections import defaultdict
import re

import torch

from qmsum_data import build_qmsum_answer_prompt, build_qmsum_evidence_answer_prompt


BAD_OUTPUT_PATTERNS = [
    ("url", re.compile(r"https?://|www\.", re.IGNORECASE)),
    ("web_ui_fragment", re.compile(r"share this highlight|minimise|minimize|read more", re.IGNORECASE)),
    ("ad_fragment", re.compile(r"\b(coupon code|promo code|order now|limited time offer)\b", re.IGNORECASE)),
    ("markdown_heading", re.compile(r"^\s*#{2,}\s+", re.MULTILINE)),
    ("template_topic_list", re.compile(r"topic:\s*(?:-\s*)+", re.IGNORECASE)),
    ("prompt_context_leak", re.compile(r"(^|\n)\s*(context|evidence|retrieved evidence|transcript)\s*:", re.IGNORECASE)),
    ("raw_turn_leak", re.compile(r"(^|\n)\s*\[turn\s+\d+\]", re.IGNORECASE)),
    ("explanation_heading", re.compile(r"(^|\n)\s*explanation\s*:", re.IGNORECASE)),
]


ANSWER_STOP_MARKERS = (
    "\nQuestion:",
    "\nQ:",
    "\nAnswer:",
    "\nFinal answer:",
    "\nContext:",
    "\nEvidence:",
    "\nRetrieved evidence:",
    "\nTranscript:",
    "\nExplanation:",
)

LEAKED_LEADING_LABEL_RE = re.compile(
    r"^\s*(context|evidence|retrieved evidence|transcript|explanation)\s*:\s*",
    re.IGNORECASE,
)


def clean_generated_answer_text(answer_text):
    text = (answer_text or "").strip()
    for marker in ANSWER_STOP_MARKERS:
        pos = text.find(marker)
        if pos > 0:
            text = text[:pos].strip()
    turn_match = re.search(r"\n\s*\[turn\s+\d+\]", text, re.IGNORECASE)
    if turn_match and turn_match.start() > 0:
        text = text[: turn_match.start()].strip()
    for prefix in ("Final answer:", "Answer:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def remove_leading_prompt_label(answer_text):
    text = (answer_text or "").strip()
    match = LEAKED_LEADING_LABEL_RE.match(text)
    if not match:
        return text, False
    return text[match.end() :].strip(), True


def collapse_repeated_answer_sentences(answer_text):
    text = (answer_text or "").strip()
    if not text:
        return text, False

    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[.!?])\s+|\n+", text)
        if piece.strip()
    ]
    if len(pieces) < 3:
        return text, False

    kept = []
    seen = set()
    changed = False
    for piece in pieces:
        normalized = re.sub(r"[^a-z0-9 ]+", "", piece.lower())
        normalized = " ".join(normalized.split())
        if len(normalized.split()) >= 5 and normalized in seen:
            changed = True
            continue
        if normalized:
            seen.add(normalized)
        kept.append(piece)

    if not changed:
        return text, False
    return " ".join(kept).strip(), True


def postprocess_generated_answer_text(answer_text):
    actions = []
    text = clean_generated_answer_text(answer_text)
    text, removed_label = remove_leading_prompt_label(text)
    if removed_label:
        actions.append("remove_leading_prompt_label")
    text, deduped = collapse_repeated_answer_sentences(text)
    if deduped:
        actions.append("collapse_repeated_sentences")
    return text.strip(), actions


def build_answer_retry_prompt(prompt_text):
    stripped = (prompt_text or "").rstrip()
    reminder = (
        "Important: write only the final answer in plain text. "
        "Do not write Context:, Evidence:, Transcript:, Question:, Answer:, "
        "or [Turn ...] lines. Do not repeat the same sentence.\n"
        "Final answer:"
    )
    for suffix in ("Final answer:", "Answer:"):
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)].rstrip() + "\n" + reminder
    return stripped + "\n\n" + reminder


def detect_bad_answer_output(answer_text):
    text = answer_text or ""
    reasons = [
        name
        for name, pattern in BAD_OUTPUT_PATTERNS
        if pattern.search(text)
    ]
    normalized = " ".join(text.strip().split()).lower()
    if normalized in {"", "explanation:", "answer:", "final answer:"}:
        reasons.append("empty_or_stub")
    if has_repetitive_answer_text(text):
        reasons.append("repetitive_answer")
    return {
        "is_bad": bool(reasons),
        "reasons": sorted(set(reasons)),
    }


def has_repetitive_answer_text(answer_text):
    text = answer_text or ""
    chunks = [
        " ".join(chunk.lower().split())
        for chunk in re.split(r"(?:[.!?]\s+|\n+)", text)
    ]
    chunks = [
        re.sub(r"[^a-z0-9 ]+", "", chunk).strip()
        for chunk in chunks
        if len(chunk.split()) >= 6
    ]
    if len(chunks) < 2:
        return False

    counts = defaultdict(int)
    for chunk in chunks:
        counts[chunk] += 1
    most_common = max(counts.values(), default=0)
    if most_common >= 3:
        return True
    if most_common >= 2 and most_common / max(1, len(chunks)) >= 0.5:
        return True
    return False


def generate_answer_text(
    model,
    tokenizer,
    prompt_text,
    max_new_tokens,
    return_metadata=False,
):
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
    raw_answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    answer_text, postprocess_actions = postprocess_generated_answer_text(raw_answer_text)

    del input_ids, attention_mask, generated
    torch.cuda.empty_cache()
    if return_metadata:
        return answer_text, {
            "raw_answer_text": raw_answer_text,
            "postprocess_actions": postprocess_actions,
        }
    return answer_text


def generate_answer_text_with_retry(model, tokenizer, prompt_text, max_new_tokens):
    answer_text, metadata = generate_answer_text(
        model,
        tokenizer,
        prompt_text,
        max_new_tokens=max_new_tokens,
        return_metadata=True,
    )
    initial_bad = detect_bad_answer_output(answer_text)
    metadata["initial_bad_output"] = bool(initial_bad["is_bad"])
    metadata["initial_bad_output_reasons"] = initial_bad["reasons"]
    metadata["retried"] = False

    if not initial_bad["is_bad"]:
        return answer_text, metadata

    retry_prompt = build_answer_retry_prompt(prompt_text)
    retry_answer, retry_metadata = generate_answer_text(
        model,
        tokenizer,
        retry_prompt,
        max_new_tokens=max_new_tokens,
        return_metadata=True,
    )
    retry_bad = detect_bad_answer_output(retry_answer)
    metadata["retried"] = True
    metadata["retry_bad_output"] = bool(retry_bad["is_bad"])
    metadata["retry_bad_output_reasons"] = retry_bad["reasons"]
    metadata["retry_postprocess_actions"] = retry_metadata.get(
        "postprocess_actions",
        [],
    )

    if not retry_bad["is_bad"]:
        metadata["used_retry"] = True
        return retry_answer, metadata

    original_len = len(normalize_answer_text(answer_text).split())
    retry_len = len(normalize_answer_text(retry_answer).split())
    if original_len == 0 and retry_len > 0:
        metadata["used_retry"] = True
        return retry_answer, metadata

    metadata["used_retry"] = False
    return answer_text, metadata


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


def build_answer_eval(
    model,
    tokenizer,
    transcripts,
    query_text,
    gold_answer,
    selected_turns,
    max_new_tokens,
    selected_answer_turns=None,
    selected_evidence_entries=None,
    selected_context_mode="turns",
    answer_prompt_style="basic",
    answer_evidence_max_entries=80,
    answer_evidence_max_chars=600,
    oracle_turns=None,
    evaluate_oracle_answer=True,
):
    if selected_answer_turns is None:
        selected_answer_turns = selected_turns

    full_prompt = build_qmsum_answer_prompt(
        transcripts,
        query_text,
        turn_indices=None,
        prompt_style=answer_prompt_style,
    )
    if selected_context_mode in ("chunks", "chunk_turns") and selected_evidence_entries:
        selected_prompt = build_qmsum_evidence_answer_prompt(
            query_text,
            selected_evidence_entries,
            prompt_style=answer_prompt_style,
            max_entries=answer_evidence_max_entries,
            max_chars_per_entry=answer_evidence_max_chars,
        )
    else:
        selected_prompt = build_qmsum_answer_prompt(
            transcripts,
            query_text,
            turn_indices=selected_answer_turns,
            prompt_style=answer_prompt_style,
        )
    oracle_prompt = None
    if evaluate_oracle_answer and oracle_turns:
        oracle_prompt = build_qmsum_answer_prompt(
            transcripts,
            query_text,
            turn_indices=oracle_turns,
            prompt_style=answer_prompt_style,
        )

    full_answer, full_generation = generate_answer_text_with_retry(
        model,
        tokenizer,
        full_prompt,
        max_new_tokens=max_new_tokens,
    )
    selected_answer, selected_generation = generate_answer_text_with_retry(
        model,
        tokenizer,
        selected_prompt,
        max_new_tokens=max_new_tokens,
    )
    oracle_answer = ""
    oracle_generation = {
        "initial_bad_output": False,
        "initial_bad_output_reasons": [],
        "retried": False,
        "used_retry": False,
        "postprocess_actions": [],
    }
    oracle_answer_available = oracle_prompt is not None
    if oracle_answer_available:
        oracle_answer, oracle_generation = generate_answer_text_with_retry(
            model,
            tokenizer,
            oracle_prompt,
            max_new_tokens=max_new_tokens,
        )

    full_answer_f1 = compute_text_f1(full_answer, gold_answer)
    selected_answer_f1 = compute_text_f1(selected_answer, gold_answer)
    oracle_answer_f1 = (
        compute_text_f1(oracle_answer, gold_answer) if oracle_answer_available else 0.0
    )
    full_bad_output = detect_bad_answer_output(full_answer)
    selected_bad_output = detect_bad_answer_output(selected_answer)
    oracle_bad_output = (
        detect_bad_answer_output(oracle_answer)
        if oracle_answer_available
        else {"is_bad": False, "reasons": []}
    )

    full_context_tokens = len(
        tokenizer(full_prompt, add_special_tokens=False).input_ids
    )
    selected_context_tokens = len(
        tokenizer(selected_prompt, add_special_tokens=False).input_ids
    )
    oracle_context_tokens = (
        len(tokenizer(oracle_prompt, add_special_tokens=False).input_ids)
        if oracle_answer_available
        else 0
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
        "oracle_answer": oracle_answer,
        "oracle_answer_available": bool(oracle_answer_available),
        "full_answer_f1": float(full_answer_f1),
        "selected_answer_f1": float(selected_answer_f1),
        "oracle_answer_f1": float(oracle_answer_f1),
        "answer_f1_delta": float(selected_answer_f1 - full_answer_f1),
        "oracle_answer_f1_delta_vs_full": (
            float(oracle_answer_f1 - full_answer_f1)
            if oracle_answer_available
            else None
        ),
        "selected_answer_f1_delta_vs_oracle": (
            float(selected_answer_f1 - oracle_answer_f1)
            if oracle_answer_available
            else None
        ),
        "full_context_tokens": int(full_context_tokens),
        "selected_context_tokens": int(selected_context_tokens),
        "oracle_context_tokens": int(oracle_context_tokens),
        "context_token_saving_ratio": float(context_token_saving_ratio),
        "selected_beats_or_matches_full": bool(selected_answer_f1 >= full_answer_f1),
        "oracle_beats_or_matches_full": (
            bool(oracle_answer_f1 >= full_answer_f1)
            if oracle_answer_available
            else None
        ),
        "selected_beats_or_matches_oracle": (
            bool(selected_answer_f1 >= oracle_answer_f1)
            if oracle_answer_available
            else None
        ),
        "selected_context_mode": selected_context_mode,
        "answer_prompt_style": answer_prompt_style,
        "full_bad_output": bool(full_bad_output["is_bad"]),
        "selected_bad_output": bool(selected_bad_output["is_bad"]),
        "oracle_bad_output": bool(oracle_bad_output["is_bad"]),
        "full_bad_output_reasons": full_bad_output["reasons"],
        "selected_bad_output_reasons": selected_bad_output["reasons"],
        "oracle_bad_output_reasons": oracle_bad_output["reasons"],
        "full_initial_bad_output": bool(full_generation.get("initial_bad_output", False)),
        "selected_initial_bad_output": bool(
            selected_generation.get("initial_bad_output", False)
        ),
        "oracle_initial_bad_output": bool(
            oracle_generation.get("initial_bad_output", False)
        ),
        "full_generation_retried": bool(full_generation.get("retried", False)),
        "selected_generation_retried": bool(selected_generation.get("retried", False)),
        "oracle_generation_retried": bool(oracle_generation.get("retried", False)),
        "full_generation_used_retry": bool(full_generation.get("used_retry", False)),
        "selected_generation_used_retry": bool(
            selected_generation.get("used_retry", False)
        ),
        "oracle_generation_used_retry": bool(oracle_generation.get("used_retry", False)),
        "full_postprocess_actions": full_generation.get("postprocess_actions", []),
        "selected_postprocess_actions": selected_generation.get(
            "postprocess_actions",
            [],
        ),
        "oracle_postprocess_actions": oracle_generation.get("postprocess_actions", []),
        "full_initial_bad_output_reasons": full_generation.get(
            "initial_bad_output_reasons",
            [],
        ),
        "selected_initial_bad_output_reasons": selected_generation.get(
            "initial_bad_output_reasons",
            [],
        ),
        "oracle_initial_bad_output_reasons": oracle_generation.get(
            "initial_bad_output_reasons",
            [],
        ),
    }
