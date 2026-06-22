#!/usr/bin/env python3
"""Analyze QMSum selective-answer bad cases from saved TSV/JSONL outputs.

This script does not run the model. It only reads saved case summaries and
optional answer logs, then writes compact reports that help identify whether
the remaining quality gap is mostly from routing evidence, gate regressions, or
answer generation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


CaseKey = Tuple[str, str]


def case_key(row: dict) -> CaseKey:
    return str(row.get("doc_id", "")), str(row.get("query_idx", ""))


def as_float(row: dict, field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict, field: str, default: int = 0) -> int:
    value = row.get(field, "")
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def bool_flag(row: dict, field: str) -> bool:
    value = str(row.get(field, "")).strip().lower()
    return value in {"1", "true", "yes"}


def split_reason_text(value) -> Set[str]:
    if value in ("", None):
        return set()
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    if not text:
        return set()
    text = text.strip("[]")
    out = set()
    for part in text.replace("'", "").replace('"', "").split(","):
        part = part.strip()
        if part:
            out.add(part)
    return out


def preview(text: str, limit: int = 260) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_answer_log(path: Optional[Path]) -> Dict[CaseKey, dict]:
    if not path or not path.exists():
        return {}
    out: Dict[CaseKey, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out[(str(item.get("doc_id", "")), str(item.get("query_idx", "")))] = item
    return out


def avg(rows: Iterable[dict], field: str) -> float:
    vals = [as_float(row, field) for row in rows]
    return sum(vals) / len(vals) if vals else 0.0


def hit_rate(rows: Iterable[dict], field: str) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(1 for row in rows if str(row.get(field, "")).strip() == "1") / len(rows)


def summarize_rows(name: str, rows: List[dict]) -> dict:
    return {
        "case": name,
        "n": len(rows),
        "full_f1": avg(rows, "full_answer_f1"),
        "selected_f1": avg(rows, "selected_answer_f1"),
        "oracle_f1": avg(rows, "oracle_answer_f1"),
        "delta": avg(rows, "answer_f1_delta"),
        "selected_oracle_gap": avg(rows, "selected_answer_f1_delta_vs_oracle"),
        "turn_recall": avg(rows, "selected_turn_recall"),
        "turn_precision": avg(rows, "selected_turn_precision"),
        "turn_f1": avg(rows, "selected_turn_f1"),
        "ctx_saving_pct": avg(rows, "ctx_token_saving_pct"),
        "qk_ms": avg(rows, "qk_scoring_ms"),
        "routing_ms": avg(rows, "routing_overhead_ms"),
        "selected_ttft_ms": avg(rows, "selected_ttft_ms"),
        "lexical_top1_hit": hit_rate(rows, "lexical_top1_hit"),
        "qk_top1_hit": hit_rate(rows, "qk_top1_hit"),
    }


def summarize_issue_rows(issue: str, rows: List[dict], total_n: int) -> dict:
    return {
        "issue": issue,
        "n": len(rows),
        "rate": (len(rows) / total_n) if total_n else 0.0,
        "full_f1": avg(rows, "full_answer_f1"),
        "selected_f1": avg(rows, "selected_answer_f1"),
        "oracle_f1": avg(rows, "oracle_answer_f1"),
        "selected_minus_full": avg(rows, "answer_f1_delta"),
        "oracle_minus_selected": -avg(rows, "selected_answer_f1_delta_vs_oracle"),
        "turn_recall": avg(rows, "selected_turn_recall"),
        "turn_precision": avg(rows, "selected_turn_precision"),
        "turn_f1": avg(rows, "selected_turn_f1"),
        "selected_bad_output_rate": hit_rate(rows, "selected_bad_output"),
        "selected_zero_f1_rate": (
            sum(1 for row in rows if as_float(row, "selected_answer_f1") == 0.0) / len(rows)
            if rows
            else 0.0
        ),
        "qk_candidates": avg(rows, "num_candidates_scored_qk"),
        "qk_ms": avg(rows, "qk_scoring_ms"),
        "selected_ttft_ms": avg(rows, "selected_ttft_ms"),
        "next_action": issue_next_action(issue),
    }


def issue_next_action(issue: str) -> str:
    mapping = {
        "topic_miss": "improve top-level topic routing or add low-cost top-2 fallback",
        "zero_turn_recall": "debug topic/chunk routing; selected evidence misses all gold turns",
        "low_turn_recall": "improve fine chunk ranking or add recall-oriented rerank/expansion",
        "large_oracle_gap": "routing/evidence selection is below gold-turn upper bound",
        "large_full_gap": "inspect whether full has evidence selected route misses",
        "selected_zero_f1": "inspect exact selected answer and evidence; often routing or prompt failure",
        "selected_bad_output": "tighten generation cleanup/prompt stop behavior",
        "prompt_context_leak": "fix prompt continuation cleanup or add stronger stop markers",
        "repetitive_answer": "reduce answer length or add repetition cleanup",
        "full_over_model_limit": "mark full baseline as less reliable for these cases",
        "low_oracle_ceiling": "answer generator/evaluation is weak even with gold turns",
        "good_or_minor": "not a priority bad case",
    }
    return mapping.get(issue, "inspect case")


def write_tsv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def enrich_row(row: dict, answer_log: Dict[CaseKey, dict]) -> dict:
    key = case_key(row)
    item = answer_log.get(key, {})
    full_f1 = as_float(row, "full_answer_f1")
    selected_f1 = as_float(row, "selected_answer_f1")
    oracle_f1 = as_float(row, "oracle_answer_f1")
    return {
        "doc_id": key[0],
        "query_idx": key[1],
        "meeting_id": row.get("meeting_id", item.get("meeting_id", "")),
        "query_budget_type": row.get("query_budget_type", item.get("query_budget_type", "")),
        "query": preview(item.get("query", "")),
        "full_f1": f"{full_f1:.4f}",
        "selected_f1": f"{selected_f1:.4f}",
        "oracle_f1": f"{oracle_f1:.4f}",
        "full_minus_selected": f"{full_f1 - selected_f1:.4f}",
        "oracle_minus_selected": f"{oracle_f1 - selected_f1:.4f}",
        "turn_recall": f"{as_float(row, 'selected_turn_recall'):.4f}",
        "turn_precision": f"{as_float(row, 'selected_turn_precision'):.4f}",
        "turn_f1": f"{as_float(row, 'selected_turn_f1'):.4f}",
        "lexical_top1_hit": row.get("lexical_top1_hit", ""),
        "qk_top1_hit": row.get("qk_top1_hit", ""),
        "selected_turn_hit": row.get("selected_turn_hit", ""),
        "selected_units": row.get("selected_units", ""),
        "relevant_units": row.get("relevant_units", ""),
        "candidate_after": row.get("num_candidates_after_prefilter", ""),
        "gate_after": row.get("coarse_segment_gate_after", ""),
        "gate_prune_pct": f"{100.0 * as_float(row, 'coarse_segment_gate_prune_ratio'):.1f}",
        "selected_ttft_ms": f"{as_float(row, 'selected_ttft_ms'):.2f}",
        "qk_ms": f"{as_float(row, 'qk_scoring_ms'):.2f}",
        "selected_bad_output": row.get("selected_bad_output", item.get("selected_bad_output", "")),
        "selected_bad_output_reasons": ",".join(
            sorted(
                split_reason_text(row.get("selected_bad_output_reasons", ""))
                | split_reason_text(item.get("selected_bad_output_reasons", []))
            )
        ),
        "full_context_tokens": item.get("full_context_tokens", ""),
        "selected_context_tokens": item.get("selected_context_tokens", ""),
        "oracle_context_tokens": item.get("oracle_context_tokens", ""),
        "gold_preview": preview(item.get("gold_answer", "")),
        "full_preview": preview(item.get("full_answer", "")),
        "selected_preview": preview(item.get("selected_answer", "")),
        "oracle_preview": preview(item.get("oracle_answer", "")),
    }


def classify_case(
    row: dict,
    item: dict,
    *,
    low_recall_threshold: float,
    large_gap_threshold: float,
    low_oracle_threshold: float,
    max_model_len: int,
    answer_max_new_tokens: int,
) -> Tuple[str, List[str]]:
    issues: List[str] = []
    full_f1 = as_float(row, "full_answer_f1")
    selected_f1 = as_float(row, "selected_answer_f1")
    oracle_f1 = as_float(row, "oracle_answer_f1")
    turn_recall = as_float(row, "selected_turn_recall")
    selected_bad = bool_flag(row, "selected_bad_output") or bool(item.get("selected_bad_output", False))
    reasons = split_reason_text(row.get("selected_bad_output_reasons", "")) | split_reason_text(
        item.get("selected_bad_output_reasons", [])
    )

    if not bool_flag(row, "lexical_top1_hit") or not bool_flag(row, "qk_top1_hit"):
        issues.append("topic_miss")
    if turn_recall <= 0.0:
        issues.append("zero_turn_recall")
    elif turn_recall < low_recall_threshold:
        issues.append("low_turn_recall")
    if selected_f1 == 0.0:
        issues.append("selected_zero_f1")
    if (full_f1 - selected_f1) >= large_gap_threshold:
        issues.append("large_full_gap")
    if (oracle_f1 - selected_f1) >= large_gap_threshold:
        issues.append("large_oracle_gap")
    if selected_bad:
        issues.append("selected_bad_output")
    if "prompt_context_leak" in reasons:
        issues.append("prompt_context_leak")
    if "repetitive_answer" in reasons:
        issues.append("repetitive_answer")
    if oracle_f1 < low_oracle_threshold:
        issues.append("low_oracle_ceiling")

    full_context_tokens = int(item.get("full_context_tokens", 0) or 0)
    if full_context_tokens and full_context_tokens + answer_max_new_tokens > max_model_len:
        issues.append("full_over_model_limit")

    if not issues:
        issues.append("good_or_minor")

    priority = [
        "prompt_context_leak",
        "repetitive_answer",
        "selected_bad_output",
        "topic_miss",
        "zero_turn_recall",
        "low_turn_recall",
        "large_oracle_gap",
        "large_full_gap",
        "selected_zero_f1",
        "full_over_model_limit",
        "low_oracle_ceiling",
        "good_or_minor",
    ]
    primary = min(issues, key=lambda issue: priority.index(issue) if issue in priority else len(priority))
    return primary, issues


def build_target_diagnostics(
    rows: List[dict],
    answer_log: Dict[CaseKey, dict],
    *,
    low_recall_threshold: float,
    large_gap_threshold: float,
    low_oracle_threshold: float,
    max_model_len: int,
    answer_max_new_tokens: int,
) -> Tuple[List[dict], List[dict]]:
    diagnostic_rows: List[dict] = []
    rows_by_issue: Dict[str, List[dict]] = {}

    for row in rows:
        key = case_key(row)
        item = answer_log.get(key, {})
        primary_issue, issues = classify_case(
            row,
            item,
            low_recall_threshold=low_recall_threshold,
            large_gap_threshold=large_gap_threshold,
            low_oracle_threshold=low_oracle_threshold,
            max_model_len=max_model_len,
            answer_max_new_tokens=answer_max_new_tokens,
        )
        for issue in issues:
            rows_by_issue.setdefault(issue, []).append(row)

        full_f1 = as_float(row, "full_answer_f1")
        selected_f1 = as_float(row, "selected_answer_f1")
        oracle_f1 = as_float(row, "oracle_answer_f1")
        reasons = split_reason_text(row.get("selected_bad_output_reasons", "")) | split_reason_text(
            item.get("selected_bad_output_reasons", [])
        )
        diagnostic_rows.append(
            {
                "doc_id": key[0],
                "query_idx": key[1],
                "meeting_id": row.get("meeting_id", item.get("meeting_id", "")),
                "primary_issue": primary_issue,
                "issues": ",".join(issues),
                "query_budget_type": row.get("query_budget_type", item.get("query_budget_type", "")),
                "query": preview(item.get("query", "")),
                "full_f1": f"{full_f1:.4f}",
                "selected_f1": f"{selected_f1:.4f}",
                "oracle_f1": f"{oracle_f1:.4f}",
                "selected_minus_full": f"{selected_f1 - full_f1:.4f}",
                "oracle_minus_selected": f"{oracle_f1 - selected_f1:.4f}",
                "turn_recall": f"{as_float(row, 'selected_turn_recall'):.4f}",
                "turn_precision": f"{as_float(row, 'selected_turn_precision'):.4f}",
                "turn_f1": f"{as_float(row, 'selected_turn_f1'):.4f}",
                "selected_turn_hit": row.get("selected_turn_hit", ""),
                "lexical_top1_hit": row.get("lexical_top1_hit", ""),
                "qk_top1_hit": row.get("qk_top1_hit", ""),
                "selected_bad_output": row.get("selected_bad_output", item.get("selected_bad_output", "")),
                "selected_bad_output_reasons": ",".join(sorted(reasons)),
                "full_context_tokens": item.get("full_context_tokens", ""),
                "selected_context_tokens": item.get("selected_context_tokens", ""),
                "oracle_context_tokens": item.get("oracle_context_tokens", ""),
                "num_candidates_scored_qk": row.get("num_candidates_scored_qk", ""),
                "selected_ttft_ms": f"{as_float(row, 'selected_ttft_ms'):.2f}",
                "qk_ms": f"{as_float(row, 'qk_scoring_ms'):.2f}",
                "selected_units": row.get("selected_units", ""),
                "relevant_units": row.get("relevant_units", ""),
                "selected_turns_preview": preview(item.get("selected_turns", []), limit=180),
                "matched_turns_preview": preview(item.get("matched_turns", []), limit=180),
                "gold_preview": preview(item.get("gold_answer", "")),
                "full_preview": preview(item.get("full_answer", "")),
                "selected_preview": preview(item.get("selected_answer", "")),
                "oracle_preview": preview(item.get("oracle_answer", "")),
            }
        )

    issue_summary = [
        summarize_issue_rows(issue, issue_rows, len(rows))
        for issue, issue_rows in sorted(
            rows_by_issue.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    ]
    diagnostic_rows.sort(
        key=lambda row: (
            float(row["selected_minus_full"]),
            -float(row["oracle_minus_selected"]),
        )
    )
    return diagnostic_rows, issue_summary


def compare_gate_to_baseline(
    baseline_rows: List[dict],
    gate_rows: List[dict],
    baseline_log: Dict[CaseKey, dict],
    gate_log: Dict[CaseKey, dict],
) -> List[dict]:
    baseline_map = {case_key(row): row for row in baseline_rows}
    out = []
    for gate_row in gate_rows:
        key = case_key(gate_row)
        base_row = baseline_map.get(key)
        if not base_row:
            continue
        base_sel = as_float(base_row, "selected_answer_f1")
        gate_sel = as_float(gate_row, "selected_answer_f1")
        merged = {
            "doc_id": key[0],
            "query_idx": key[1],
            "meeting_id": gate_row.get("meeting_id", ""),
            "query_budget_type": gate_row.get("query_budget_type", ""),
            "query": preview(gate_log.get(key, {}).get("query", baseline_log.get(key, {}).get("query", ""))),
            "baseline_selected_f1": f"{base_sel:.4f}",
            "gate_selected_f1": f"{gate_sel:.4f}",
            "gate_minus_baseline_selected": f"{gate_sel - base_sel:.4f}",
            "baseline_full_f1": f"{as_float(base_row, 'full_answer_f1'):.4f}",
            "gate_full_f1": f"{as_float(gate_row, 'full_answer_f1'):.4f}",
            "baseline_turn_recall": f"{as_float(base_row, 'selected_turn_recall'):.4f}",
            "gate_turn_recall": f"{as_float(gate_row, 'selected_turn_recall'):.4f}",
            "baseline_qk_ms": f"{as_float(base_row, 'qk_scoring_ms'):.2f}",
            "gate_qk_ms": f"{as_float(gate_row, 'qk_scoring_ms'):.2f}",
            "baseline_ttft_ms": f"{as_float(base_row, 'selected_ttft_ms'):.2f}",
            "gate_ttft_ms": f"{as_float(gate_row, 'selected_ttft_ms'):.2f}",
            "gate_prune_pct": f"{100.0 * as_float(gate_row, 'coarse_segment_gate_prune_ratio'):.1f}",
            "gold_preview": preview(gate_log.get(key, {}).get("gold_answer", "")),
            "baseline_selected_preview": preview(baseline_log.get(key, {}).get("selected_answer", "")),
            "gate_selected_preview": preview(gate_log.get(key, {}).get("selected_answer", "")),
        }
        out.append(merged)
    out.sort(key=lambda row: float(row["gate_minus_baseline_selected"]))
    return out


def write_markdown(
    path: Path,
    summary: List[dict],
    bad_full: List[dict],
    bad_oracle: List[dict],
    regressions: List[dict],
    diagnostic_rows: Optional[List[dict]],
    issue_summary: Optional[List[dict]],
    top_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# QMSum Bad Case Analysis\n\n")
        f.write("## Summary\n\n")
        f.write("| case | n | full_f1 | selected_f1 | oracle_f1 | delta | qk_ms | selected_ttft_ms |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            f.write(
                "| {case} | {n} | {full_f1:.4f} | {selected_f1:.4f} | "
                "{oracle_f1:.4f} | {delta:.4f} | {qk_ms:.2f} | "
                "{selected_ttft_ms:.2f} |\n".format(**row)
            )
        f.write("\n")

        if issue_summary is not None:
            f.write("## Target Diagnosis\n\n")
            f.write(
                "| issue | n | rate | selected_f1 | full_f1 | oracle_f1 | "
                "turn_recall | bad_rate | next_action |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
            for row in issue_summary:
                f.write(
                    "| {issue} | {n} | {rate:.1%} | {selected_f1:.4f} | "
                    "{full_f1:.4f} | {oracle_f1:.4f} | {turn_recall:.4f} | "
                    "{selected_bad_output_rate:.1%} | {next_action} |\n".format(**row)
                )
            f.write("\n")

        if diagnostic_rows is not None:
            f.write("## Highest-Priority Target Cases\n\n")
            for row in diagnostic_rows[:top_n]:
                f.write(f"### doc {row.get('doc_id')} / query {row.get('query_idx')}\n\n")
                for field in [
                    "primary_issue",
                    "issues",
                    "query",
                    "selected_minus_full",
                    "oracle_minus_selected",
                    "full_f1",
                    "selected_f1",
                    "oracle_f1",
                    "turn_recall",
                    "lexical_top1_hit",
                    "qk_top1_hit",
                    "selected_bad_output_reasons",
                    "gold_preview",
                    "selected_preview",
                    "full_preview",
                    "oracle_preview",
                ]:
                    value = row.get(field, "")
                    if value != "":
                        f.write(f"- {field}: {value}\n")
                f.write("\n")

        def section(title: str, rows: List[dict], fields: List[str]) -> None:
            f.write(f"## {title}\n\n")
            for row in rows[:top_n]:
                f.write(f"### doc {row.get('doc_id')} / query {row.get('query_idx')}\n\n")
                for field in fields:
                    value = row.get(field, "")
                    if value != "":
                        f.write(f"- {field}: {value}\n")
                f.write("\n")

        section(
            "Worst Selected vs Full",
            bad_full,
            [
                "query",
                "full_minus_selected",
                "full_f1",
                "selected_f1",
                "oracle_f1",
                "turn_recall",
                "lexical_top1_hit",
                "qk_top1_hit",
                "gold_preview",
                "selected_preview",
                "full_preview",
            ],
        )
        section(
            "Largest Selected vs Oracle Gap",
            bad_oracle,
            [
                "query",
                "oracle_minus_selected",
                "selected_f1",
                "oracle_f1",
                "turn_recall",
                "selected_turn_hit",
                "gold_preview",
                "selected_preview",
                "oracle_preview",
            ],
        )
        section(
            "Gate Regressions vs Baseline",
            regressions,
            [
                "query",
                "gate_minus_baseline_selected",
                "baseline_selected_f1",
                "gate_selected_f1",
                "baseline_turn_recall",
                "gate_turn_recall",
                "baseline_qk_ms",
                "gate_qk_ms",
                "gate_prune_pct",
                "gold_preview",
                "baseline_selected_preview",
                "gate_selected_preview",
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-tsv", type=Path, default=None)
    parser.add_argument("--gate-tsv", type=Path, default=None)
    parser.add_argument("--baseline-answer-log", type=Path, default=None)
    parser.add_argument("--gate-answer-log", type=Path, default=None)
    parser.add_argument("--target-tsv", type=Path, default=None)
    parser.add_argument("--target-answer-log", type=Path, default=None)
    parser.add_argument("--target-name", type=str, default="")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/qmsum_bad_case_analysis"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--low-recall-threshold", type=float, default=0.10)
    parser.add_argument("--large-gap-threshold", type=float, default=0.10)
    parser.add_argument("--low-oracle-threshold", type=float, default=0.12)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--answer-max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    if not args.target_tsv and not args.gate_tsv:
        parser.error("Provide --target-tsv or --gate-tsv")

    baseline_rows = read_tsv(args.baseline_tsv) if args.baseline_tsv else []
    gate_rows = read_tsv(args.gate_tsv) if args.gate_tsv else []
    baseline_log = read_answer_log(args.baseline_answer_log)
    gate_log = read_answer_log(args.gate_answer_log)
    target_tsv = args.target_tsv or args.gate_tsv
    target_log_path = args.target_answer_log or args.gate_answer_log
    target_rows = read_tsv(target_tsv) if target_tsv else []
    target_log = read_answer_log(target_log_path)
    target_name = args.target_name or ("gate" if args.gate_tsv else "target")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    if baseline_rows:
        summary.append(summarize_rows("baseline", baseline_rows))
    if gate_rows:
        summary.append(summarize_rows("gate", gate_rows))
    elif target_rows:
        summary.append(summarize_rows(target_name, target_rows))
    write_tsv(
        args.out_dir / "summary.tsv",
        [
            {
                key: (f"{value:.4f}" if isinstance(value, float) else value)
                for key, value in row.items()
            }
            for row in summary
        ],
        [
            "case",
            "n",
            "full_f1",
            "selected_f1",
            "oracle_f1",
            "delta",
            "selected_oracle_gap",
            "turn_recall",
            "turn_precision",
            "turn_f1",
            "ctx_saving_pct",
            "qk_ms",
            "routing_ms",
            "selected_ttft_ms",
            "lexical_top1_hit",
            "qk_top1_hit",
        ],
    )

    diagnostic_rows, issue_summary = build_target_diagnostics(
        target_rows,
        target_log,
        low_recall_threshold=args.low_recall_threshold,
        large_gap_threshold=args.large_gap_threshold,
        low_oracle_threshold=args.low_oracle_threshold,
        max_model_len=args.max_model_len,
        answer_max_new_tokens=args.answer_max_new_tokens,
    )
    diagnostic_fields = [
        "doc_id",
        "query_idx",
        "meeting_id",
        "primary_issue",
        "issues",
        "query_budget_type",
        "query",
        "full_f1",
        "selected_f1",
        "oracle_f1",
        "selected_minus_full",
        "oracle_minus_selected",
        "turn_recall",
        "turn_precision",
        "turn_f1",
        "selected_turn_hit",
        "lexical_top1_hit",
        "qk_top1_hit",
        "selected_bad_output",
        "selected_bad_output_reasons",
        "full_context_tokens",
        "selected_context_tokens",
        "oracle_context_tokens",
        "num_candidates_scored_qk",
        "selected_ttft_ms",
        "qk_ms",
        "selected_units",
        "relevant_units",
        "selected_turns_preview",
        "matched_turns_preview",
        "gold_preview",
        "full_preview",
        "selected_preview",
        "oracle_preview",
    ]
    write_tsv(args.out_dir / "target_diagnostic_cases.tsv", diagnostic_rows, diagnostic_fields)
    write_tsv(
        args.out_dir / "issue_summary.tsv",
        [
            {
                key: (f"{value:.4f}" if isinstance(value, float) else value)
                for key, value in row.items()
            }
            for row in issue_summary
        ],
        [
            "issue",
            "n",
            "rate",
            "full_f1",
            "selected_f1",
            "oracle_f1",
            "selected_minus_full",
            "oracle_minus_selected",
            "turn_recall",
            "turn_precision",
            "turn_f1",
            "selected_bad_output_rate",
            "selected_zero_f1_rate",
            "qk_candidates",
            "qk_ms",
            "selected_ttft_ms",
            "next_action",
        ],
    )
    write_tsv(
        args.out_dir / "zero_recall_cases.tsv",
        [row for row in diagnostic_rows if "zero_turn_recall" in row["issues"]][: args.top_n],
        diagnostic_fields,
    )
    write_tsv(
        args.out_dir / "prompt_leak_cases.tsv",
        [row for row in diagnostic_rows if "prompt_context_leak" in row["issues"]][: args.top_n],
        diagnostic_fields,
    )
    write_tsv(
        args.out_dir / "large_oracle_gap_cases.tsv",
        [row for row in diagnostic_rows if "large_oracle_gap" in row["issues"]][: args.top_n],
        diagnostic_fields,
    )

    enriched_gate = [enrich_row(row, target_log) for row in target_rows]
    bad_full = sorted(
        enriched_gate,
        key=lambda row: float(row["full_minus_selected"]),
        reverse=True,
    )
    bad_oracle = sorted(
        enriched_gate,
        key=lambda row: float(row["oracle_minus_selected"]),
        reverse=True,
    )
    regressions = []
    if baseline_rows and gate_rows:
        regressions = compare_gate_to_baseline(
            baseline_rows,
            gate_rows,
            baseline_log,
            gate_log,
        )

    common_fields = [
        "doc_id",
        "query_idx",
        "meeting_id",
        "query_budget_type",
        "query",
        "full_f1",
        "selected_f1",
        "oracle_f1",
        "full_minus_selected",
        "oracle_minus_selected",
        "turn_recall",
        "turn_precision",
        "turn_f1",
        "lexical_top1_hit",
        "qk_top1_hit",
        "selected_turn_hit",
        "selected_units",
        "relevant_units",
        "candidate_after",
        "gate_after",
        "gate_prune_pct",
        "selected_ttft_ms",
        "qk_ms",
        "selected_bad_output",
        "selected_bad_output_reasons",
        "full_context_tokens",
        "selected_context_tokens",
        "oracle_context_tokens",
        "gold_preview",
        "full_preview",
        "selected_preview",
        "oracle_preview",
    ]
    write_tsv(args.out_dir / "worst_selected_vs_full.tsv", bad_full[: args.top_n], common_fields)
    write_tsv(args.out_dir / "worst_selected_vs_oracle.tsv", bad_oracle[: args.top_n], common_fields)
    write_tsv(
        args.out_dir / "gate_regressions_vs_baseline.tsv",
        regressions[: args.top_n],
        [
            "doc_id",
            "query_idx",
            "meeting_id",
            "query_budget_type",
            "query",
            "baseline_selected_f1",
            "gate_selected_f1",
            "gate_minus_baseline_selected",
            "baseline_full_f1",
            "gate_full_f1",
            "baseline_turn_recall",
            "gate_turn_recall",
            "baseline_qk_ms",
            "gate_qk_ms",
            "baseline_ttft_ms",
            "gate_ttft_ms",
            "gate_prune_pct",
            "gold_preview",
            "baseline_selected_preview",
            "gate_selected_preview",
        ],
    )
    write_markdown(
        args.out_dir / "bad_case_report.md",
        summary,
        bad_full,
        bad_oracle,
        regressions,
        diagnostic_rows,
        issue_summary,
        min(args.top_n, 10),
    )

    print(f"Saved summary to {args.out_dir / 'summary.tsv'}")
    print(f"Saved issue summary to {args.out_dir / 'issue_summary.tsv'}")
    print(f"Saved target diagnostics to {args.out_dir / 'target_diagnostic_cases.tsv'}")
    print(f"Saved bad case report to {args.out_dir / 'bad_case_report.md'}")


if __name__ == "__main__":
    main()
