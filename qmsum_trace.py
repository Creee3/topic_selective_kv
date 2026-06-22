import json
import os

import numpy as np


def preview_text(text, limit=160):
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def should_trace_case(args, doc_id, query_idx):
    return args.trace_doc_id == doc_id and args.trace_query_idx == query_idx


def write_case_trace(trace_payload, output_dir, output_format):
    os.makedirs(output_dir, exist_ok=True)
    doc_id = int(trace_payload["target"]["doc_id"])
    query_idx = int(trace_payload["target"]["query_idx"])
    prefix = os.path.join(output_dir, f"doc_{doc_id}_query_{query_idx}")

    json_path = None
    md_path = None

    if output_format in ["json", "both"]:
        json_path = prefix + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(trace_payload), f, indent=2, ensure_ascii=False)

    md_lines = [
        "# QMSum Single-Case Trace",
        "",
        f"- doc_id: `{doc_id}`",
        f"- query_idx: `{query_idx}`",
        f"- meeting_id: `{trace_payload['meeting'].get('meeting_id', '')}`",
        f"- routing_granularity: `{trace_payload['config'].get('routing_granularity', '')}`",
        "",
        "## Query",
        "",
        f"- text: `{trace_payload['query'].get('text', '')}`",
        f"- relevant_turn_count: `{trace_payload['query'].get('relevant_turn_count', 0)}`",
        f"- relevant_topic_ids: `{trace_payload['query'].get('relevant_topic_ids', [])}`",
        "",
        "## Functions Used",
        "",
    ]

    for step in trace_payload.get("functions_used", []):
        md_lines.append(f"1. `{step['function']}`: {step['purpose']}")
        if step.get("key_outputs"):
            md_lines.append(f"   outputs: `{step['key_outputs']}`")

    md_lines.extend(["", "## Selected Topics", ""])
    for topic in trace_payload.get("top_level_routing", {}).get("selected_topics", []):
        md_lines.append(
            f"- T{topic['topic_id']}: `{topic['label']}` | selection_score={topic['selection_score']:.4f} | num_turns={topic['num_turns']}"
        )

    md_lines.extend(["", "## Selected Chunks", ""])
    for chunk in trace_payload.get("chunk_routing", {}).get("selected_candidates_preview", []):
        md_lines.append(
            f"- turn={chunk['turn_idx']}, chunk={chunk['local_chunk_idx']}, topic_ids={chunk['topic_ids']}, score={chunk['score']:.4f}, tokens=[{chunk['start_t']}, {chunk['end_t']})"
        )

    eval_info = trace_payload.get("evaluation", {})
    md_lines.extend(
        [
            "",
            "## Evaluation",
            "",
            f"- selected_topic_hit: `{eval_info.get('selected_topic_hit', False)}`",
            f"- selected_turn_hit: `{eval_info.get('selected_turn_hit', False)}`",
            f"- recall: `{eval_info.get('turn_recall', 0.0):.4f}`",
            f"- precision: `{eval_info.get('turn_precision', 0.0):.4f}`",
            f"- f1: `{eval_info.get('turn_f1', 0.0):.4f}`",
        ]
    )

    answer_eval = trace_payload.get("answer_eval")
    if answer_eval:
        md_lines.extend(
            [
                "",
                "## Answer Eval",
                "",
                f"- gold_answer: `{answer_eval.get('gold_answer_preview', '')}`",
                f"- full_answer: `{answer_eval.get('full_answer_preview', '')}`",
                f"- selected_answer: `{answer_eval.get('selected_answer_preview', '')}`",
                f"- full_answer_f1: `{answer_eval.get('full_answer_f1', 0.0):.4f}`",
                f"- selected_answer_f1: `{answer_eval.get('selected_answer_f1', 0.0):.4f}`",
                f"- answer_f1_delta: `{answer_eval.get('answer_f1_delta', 0.0):.4f}`",
                f"- full_context_tokens: `{answer_eval.get('full_context_tokens', 0)}`",
                f"- selected_context_tokens: `{answer_eval.get('selected_context_tokens', 0)}`",
                f"- context_token_saving_ratio: `{answer_eval.get('context_token_saving_ratio', 0.0):.4f}`",
            ]
        )

    if output_format in ["md", "both"]:
        md_path = prefix + ".md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

    return json_path, md_path
