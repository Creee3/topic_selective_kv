import json
import os


def _as_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_output_suffix(args):
    tag = getattr(args, "case_summary_tag", "")
    if tag:
        return f"N{args.num_nodes}_{args.start_doc}_{args.end_doc}_{tag}"
    return f"N{args.num_nodes}_{args.start_doc}_{args.end_doc}"


def get_prefilter_fields(result):
    qk_route_info = result.get("qk_route_info") or {}
    routing_eval = result.get("routing_eval") or {}
    return {
        "mode": qk_route_info.get(
            "candidate_prefilter_mode",
            routing_eval.get("route_candidate_prefilter_mode", ""),
        ),
        "pool_size": qk_route_info.get(
            "candidate_prefilter_pool_size",
            routing_eval.get("route_candidate_prefilter_pool_size", ""),
        ),
        "requested_pool_size": qk_route_info.get(
            "candidate_prefilter_requested_pool_size",
            routing_eval.get("route_candidate_prefilter_requested_pool_size", ""),
        ),
        "prune_ratio": qk_route_info.get(
            "candidate_prefilter_prune_ratio",
            routing_eval.get("route_candidate_prefilter_prune_ratio", ""),
        ),
        "min_prune_ratio": qk_route_info.get(
            "candidate_prefilter_min_prune_ratio",
            routing_eval.get("route_candidate_prefilter_min_prune_ratio", ""),
        ),
        "keep_ratio": qk_route_info.get(
            "candidate_prefilter_keep_ratio",
            routing_eval.get("route_candidate_prefilter_keep_ratio", ""),
        ),
        "skip_reason": qk_route_info.get(
            "candidate_prefilter_skip_reason",
            routing_eval.get("route_candidate_prefilter_skip_reason", ""),
        ),
        "before_count": qk_route_info.get(
            "num_candidates_before_prefilter",
            routing_eval.get("route_num_candidates_before_prefilter", ""),
        ),
        "after_count": qk_route_info.get(
            "num_candidates_after_prefilter",
            routing_eval.get("route_num_candidates_after_prefilter", ""),
        ),
    }


def get_coarse_segment_gate_fields(result):
    qk_route_info = result.get("qk_route_info") or {}
    routing_eval = result.get("routing_eval") or {}
    return {
        "mode": qk_route_info.get(
            "route_coarse_segment_gate",
            routing_eval.get("route_coarse_segment_gate", "none"),
        ),
        "before_count": qk_route_info.get(
            "coarse_segment_gate_before",
            routing_eval.get("route_coarse_segment_gate_before", ""),
        ),
        "after_count": qk_route_info.get(
            "coarse_segment_gate_after",
            routing_eval.get("route_coarse_segment_gate_after", ""),
        ),
        "prune_ratio": qk_route_info.get(
            "coarse_segment_gate_prune_ratio",
            routing_eval.get("route_coarse_segment_gate_prune_ratio", ""),
        ),
        "num_segments": qk_route_info.get("coarse_segment_gate_num_segments", ""),
        "keep_segments": qk_route_info.get("coarse_segment_gate_keep_segments", ""),
        "preview": qk_route_info.get("coarse_segment_gate_preview", []),
    }


def _strategy_hit(routing_eval, strategy_name, field):
    return bool((routing_eval.get("strategy_hits") or {}).get(strategy_name, {}).get(field, False))


def _build_bad_case_categories(result):
    routing_eval = result.get("routing_eval") or {}
    answer_eval = result.get("answer_eval") or {}
    categories = []

    if answer_eval.get("selected_bad_output", False):
        categories.append("selected_bad_output")
    if answer_eval.get("selected_initial_bad_output", False):
        categories.append("selected_initial_bad_output")
    if not routing_eval.get("route_selected_unit_hit", False):
        categories.append("topic_miss")
    if not routing_eval.get("route_selected_turn_hit", False):
        categories.append("turn_miss")
    if _as_float(routing_eval.get("route_selected_turn_recall")) <= 0.0:
        categories.append("zero_turn_recall")
    if _as_float(answer_eval.get("answer_f1_delta")) <= -0.10:
        categories.append("large_selected_full_gap")
    if _as_float(answer_eval.get("selected_answer_f1_delta_vs_oracle")) <= -0.10:
        categories.append("large_selected_oracle_gap")
    if (
        _as_float(answer_eval.get("oracle_answer_f1_delta_vs_full")) >= 0.10
        and _as_float(answer_eval.get("selected_answer_f1_delta_vs_oracle")) <= -0.10
    ):
        categories.append("routing_quality_gap")

    return sorted(set(categories)) or ["ok"]


def write_case_summary_tsv(results, out_path):
    strategy_names = sorted(results[0].get("strategy_results", {}).keys()) if results else []
    header = [
        "doc_id",
        "query_idx",
        "meeting_id",
        "routing_unit_type",
        "relevant_units",
        "dominant_unit",
        "selected_unit_strategy",
        "selected_units",
        "selected_virtual_nodes",
        "selected_turn_hit",
        "selected_turn_recall",
        "selected_turn_precision",
        "selected_turn_f1",
        "survival_failure_stage",
        "survival_first_drop_stage",
        "survival_first_zero_stage",
        "survival_topic_filter_turn_recall",
        "survival_prefilter_turn_recall",
        "survival_dynamic_pool_turn_recall",
        "survival_coarse_gate_turn_recall",
        "survival_qk_scored_turn_recall",
        "survival_qk_selected_turn_recall",
        "query_budget_type",
        "effective_route_top_k",
        "route_selection_mode",
        "route_hybrid_core_ratio",
        "route_hybrid_core_max_per_turn",
        "candidate_prefilter_mode",
        "candidate_prefilter_pool_size",
        "candidate_prefilter_requested_pool_size",
        "candidate_prefilter_prune_ratio",
        "candidate_prefilter_keep_ratio",
        "candidate_prefilter_min_prune_ratio",
        "candidate_prefilter_skip_reason",
        "dynamic_candidate_pool_budget",
        "dynamic_candidate_pool_target",
        "dynamic_candidate_pool_prune_ratio",
        "num_candidates_before_dynamic_pool",
        "num_candidates_after_dynamic_pool",
        "num_candidates_after_coarse_segment_gate",
        "coarse_segment_gate_mode",
        "coarse_segment_gate_before",
        "coarse_segment_gate_after",
        "coarse_segment_gate_prune_ratio",
        "num_candidates_before_prefilter",
        "num_candidates_after_prefilter",
        "num_candidates_scored_qk",
        "answer_evidence_order",
        "transfer_virtual_node_count",
        "transfer_virtual_node_segments",
        "timing_is_first_query",
        "routing_overhead_ms",
        "routing_wall_clock_ms",
        "routing_simulator_excluded_ms",
        "query_tokenize_ms",
        "coarse_topic_routing_ms",
        "topic_filter_ms",
        "candidate_prefilter_ms",
        "dynamic_candidate_pool_ms",
        "coarse_segment_gate_ms",
        "candidate_key_prepare_ms",
        "query_q_prepare_ms",
        "qk_model_inference_ms",
        "qk_score_aggregation_ms",
        "qk_scoring_ms",
        "qk_total_stage_ms",
        "selection_postprocess_ms",
        "route_unaccounted_ms",
        "selected_kv_mib",
        "full_kv_mib",
        "kv_saving_pct",
        "selected_fetch_latency_ms",
        "full_fetch_latency_ms",
        "fetch_latency_saving_pct",
        "selected_ttft_ms",
        "full_ttft_ms",
        "ttft_saving_pct",
        "cachegen_full_status",
        "cachegen_full_compressed_mib",
        "cachegen_full_compression_saving_pct",
        "cachegen_full_encode_ms",
        "cachegen_full_fetch_latency_ms",
        "cachegen_full_ttft_ms",
        "selected_vs_cachegen_ttft_delta_ms",
        "selected_vs_cachegen_ttft_saving_pct",
        "cachegen_estimated_answer_f1",
        "selected_answer_f1_delta_vs_cachegen",
        "full_answer_f1",
        "selected_answer_f1",
        "oracle_answer_f1",
        "answer_f1_delta",
        "oracle_answer_f1_delta_vs_full",
        "selected_answer_f1_delta_vs_oracle",
        "full_bad_output",
        "selected_bad_output",
        "oracle_bad_output",
        "selected_bad_output_reasons",
        "selected_initial_bad_output",
        "selected_generation_retried",
        "selected_generation_used_retry",
        "selected_postprocess_actions",
        "bad_case_category",
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

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for result in results:
            routing_eval = result.get("routing_eval") or {}
            answer_eval = result.get("answer_eval") or {}
            qk_route_info = result.get("qk_route_info") or {}
            survival = qk_route_info.get("relevant_candidate_survival") or {}
            prefilter_fields = get_prefilter_fields(result)
            gate_fields = get_coarse_segment_gate_fields(result)
            system_cost = result.get("system_cost") or {}
            routing_timing = result.get("routing_timing") or {}
            selected_cost = system_cost.get("selected", {})
            full_cost = system_cost.get("full", {})
            reduction = system_cost.get("reduction", {})
            cachegen_full = result.get("cachegen_full_estimate") or {}
            bad_case_categories = _build_bad_case_categories(result)
            selected_units = qk_route_info.get("selected_topic_ids")
            if selected_units is None:
                selected_units = qk_route_info.get("selected_nodes", [])

            row = [
                str(_as_int(result.get("doc_id", -1), -1)),
                str(_as_int(result.get("query_idx", -1), -1)),
                str(result.get("meeting_id", "")),
                str(result.get("routing_unit_type", "")),
                ",".join(str(_as_int(x)) for x in result.get("relevant_nodes", [])),
                str(_as_int(result.get("dominant_relevant_node", -1), -1)),
                str(routing_eval.get("selected_unit_strategy", "")),
                ",".join(str(_as_int(x)) for x in (selected_units or [])),
                ",".join(
                    str(_as_int(x))
                    for x in qk_route_info.get("selected_virtual_node_ids", [])
                ),
                "1" if routing_eval.get("route_selected_turn_hit", False) else "0",
                f"{_as_float(routing_eval.get('route_selected_turn_recall')):.4f}",
                f"{_as_float(routing_eval.get('route_selected_turn_precision')):.4f}",
                f"{_as_float(routing_eval.get('route_selected_turn_f1')):.4f}",
                str(survival.get("failure_stage", "")),
                str(survival.get("first_drop_stage", "")),
                str(survival.get("first_zero_stage", "")),
                f"{_as_float(survival.get('topic_filter_turn_recall')):.4f}",
                f"{_as_float(survival.get('candidate_prefilter_turn_recall')):.4f}",
                f"{_as_float(survival.get('dynamic_candidate_pool_turn_recall')):.4f}",
                f"{_as_float(survival.get('coarse_segment_gate_turn_recall')):.4f}",
                f"{_as_float(survival.get('qk_scored_turn_recall')):.4f}",
                f"{_as_float(survival.get('qk_selected_turn_recall')):.4f}",
                str(qk_route_info.get("query_budget_type", "")),
                str(qk_route_info.get("effective_route_top_k", "")),
                str(qk_route_info.get("route_selection_mode", "chunk_topk")),
                str(qk_route_info.get("route_hybrid_core_ratio", "")),
                str(qk_route_info.get("route_hybrid_core_max_per_turn", "")),
                str(prefilter_fields.get("mode", "")),
                str(prefilter_fields.get("pool_size", "")),
                str(prefilter_fields.get("requested_pool_size", "")),
                str(prefilter_fields.get("prune_ratio", "")),
                str(prefilter_fields.get("keep_ratio", "")),
                str(prefilter_fields.get("min_prune_ratio", "")),
                str(prefilter_fields.get("skip_reason", "")),
                "1" if qk_route_info.get("dynamic_candidate_pool_budget", False) else "0",
                str(qk_route_info.get("dynamic_candidate_pool_target", "")),
                str(qk_route_info.get("dynamic_candidate_pool_prune_ratio", "")),
                str(qk_route_info.get("num_candidates_before_dynamic_pool", "")),
                str(qk_route_info.get("num_candidates_after_dynamic_pool", "")),
                str(qk_route_info.get("num_candidates_after_coarse_segment_gate", "")),
                str(gate_fields.get("mode", "")),
                str(gate_fields.get("before_count", "")),
                str(gate_fields.get("after_count", "")),
                str(gate_fields.get("prune_ratio", "")),
                str(prefilter_fields.get("before_count", "")),
                str(prefilter_fields.get("after_count", "")),
                str(qk_route_info.get("num_candidates_scored_qk", "")),
                str(qk_route_info.get("answer_evidence_order", "")),
                str(_as_int(routing_eval.get("route_transfer_virtual_node_count"))),
                str(_as_int(routing_eval.get("route_transfer_virtual_node_segment_count"))),
                "1" if routing_timing.get("timing_is_first_query", False) else "0",
                f"{_as_float(system_cost.get('routing_overhead_ms')):.2f}",
                f"{_as_float(system_cost.get('routing_wall_clock_ms', routing_timing.get('online_route_decision_ms'))):.2f}",
                f"{_as_float(system_cost.get('routing_simulator_excluded_ms')):.2f}",
                f"{_as_float(routing_timing.get('query_tokenize_ms')):.2f}",
                f"{_as_float(routing_timing.get('coarse_topic_routing_ms')):.2f}",
                f"{_as_float(routing_timing.get('topic_filter_ms')):.2f}",
                f"{_as_float(routing_timing.get('candidate_prefilter_ms')):.2f}",
                f"{_as_float(routing_timing.get('dynamic_candidate_pool_ms')):.2f}",
                f"{_as_float(routing_timing.get('coarse_segment_gate_ms')):.2f}",
                f"{_as_float(routing_timing.get('candidate_key_prepare_ms')):.2f}",
                f"{_as_float(routing_timing.get('query_q_prepare_ms')):.2f}",
                f"{_as_float(routing_timing.get('qk_model_inference_ms')):.2f}",
                f"{_as_float(routing_timing.get('qk_score_aggregation_ms')):.2f}",
                f"{_as_float(routing_timing.get('qk_scoring_ms')):.2f}",
                f"{_as_float(routing_timing.get('qk_total_stage_ms')):.2f}",
                f"{_as_float(routing_timing.get('selection_postprocess_ms')):.2f}",
                f"{_as_float(routing_timing.get('route_unaccounted_ms')):.2f}",
                f"{_as_float(selected_cost.get('kv_mib')):.2f}",
                f"{_as_float(full_cost.get('kv_mib')):.2f}",
                f"{100.0 * _as_float(reduction.get('kv_bytes_reduction_ratio')):.1f}",
                f"{_as_float(selected_cost.get('fetch_latency_ms')):.2f}",
                f"{_as_float(full_cost.get('fetch_latency_ms')):.2f}",
                f"{100.0 * _as_float(reduction.get('fetch_latency_reduction_ratio')):.1f}",
                f"{_as_float(selected_cost.get('estimated_ttft_ms')):.2f}",
                f"{_as_float(full_cost.get('estimated_ttft_ms')):.2f}",
                f"{100.0 * _as_float(reduction.get('ttft_reduction_ratio')):.1f}",
                str(cachegen_full.get("status", "")),
                f"{_as_float(cachegen_full.get('compressed_mib')):.2f}",
                f"{100.0 * _as_float(cachegen_full.get('compression_saving_ratio')):.1f}",
                f"{_as_float(cachegen_full.get('total_encode_ms')):.2f}",
                f"{_as_float(cachegen_full.get('fetch_latency_ms')):.2f}",
                f"{_as_float(cachegen_full.get('estimated_ttft_ms')):.2f}",
                f"{_as_float(cachegen_full.get('selected_vs_cachegen_ttft_delta_ms')):.2f}",
                f"{100.0 * _as_float(cachegen_full.get('selected_vs_cachegen_ttft_saving_ratio')):.1f}",
                f"{_as_float(cachegen_full.get('estimated_answer_f1')):.4f}",
                f"{_as_float(cachegen_full.get('selected_answer_f1_delta_vs_cachegen')):.4f}",
                f"{_as_float(answer_eval.get('full_answer_f1')):.4f}",
                f"{_as_float(answer_eval.get('selected_answer_f1')):.4f}",
                f"{_as_float(answer_eval.get('oracle_answer_f1')):.4f}",
                f"{_as_float(answer_eval.get('answer_f1_delta')):.4f}",
                f"{_as_float(answer_eval.get('oracle_answer_f1_delta_vs_full')):.4f}",
                f"{_as_float(answer_eval.get('selected_answer_f1_delta_vs_oracle')):.4f}",
                "1" if answer_eval.get("full_bad_output", False) else "0",
                "1" if answer_eval.get("selected_bad_output", False) else "0",
                "1" if answer_eval.get("oracle_bad_output", False) else "0",
                ",".join(str(x) for x in answer_eval.get("selected_bad_output_reasons", [])),
                "1" if answer_eval.get("selected_initial_bad_output", False) else "0",
                "1" if answer_eval.get("selected_generation_retried", False) else "0",
                "1" if answer_eval.get("selected_generation_used_retry", False) else "0",
                ",".join(
                    str(x)
                    for x in answer_eval.get("selected_postprocess_actions", [])
                ),
                ",".join(str(x) for x in bad_case_categories),
                f"{100.0 * _as_float(answer_eval.get('context_token_saving_ratio')):.1f}",
            ]
            for strategy_name in strategy_names:
                sr = result.get("strategy_results", {}).get(strategy_name, {})
                ranked = sr.get("ranked_nodes", [])
                row.extend(
                    [
                        str(sr.get("top_node", "")),
                        ",".join(str(x) for x in ranked[:2]),
                        "1"
                        if _strategy_hit(routing_eval, strategy_name, "top1_any_relevant_hit")
                        else "0",
                        "1"
                        if _strategy_hit(routing_eval, strategy_name, "top2_any_relevant_hit")
                        else "0",
                    ]
                )
            f.write("\t".join(row) + "\n")


def write_case_answer_log(results, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for result in results:
            answer_eval = result.get("answer_eval")
            if not answer_eval:
                continue
            qk_route_info = result.get("qk_route_info") or {}
            routing_eval = result.get("routing_eval") or {}
            route_info = result.get("route_unit_info") or {}
            payload = {
                "doc_id": result.get("doc_id"),
                "query_idx": result.get("query_idx"),
                "meeting_id": result.get("meeting_id"),
                "query": result.get("query"),
                "gold_answer": answer_eval.get("gold_answer"),
                "full_answer": answer_eval.get("full_answer"),
                "selected_answer": answer_eval.get("selected_answer"),
                "oracle_answer": answer_eval.get("oracle_answer"),
                "full_answer_f1": answer_eval.get("full_answer_f1"),
                "selected_answer_f1": answer_eval.get("selected_answer_f1"),
                "oracle_answer_f1": answer_eval.get("oracle_answer_f1"),
                "answer_f1_delta": answer_eval.get("answer_f1_delta"),
                "oracle_answer_f1_delta_vs_full": answer_eval.get(
                    "oracle_answer_f1_delta_vs_full"
                ),
                "selected_answer_f1_delta_vs_oracle": answer_eval.get(
                    "selected_answer_f1_delta_vs_oracle"
                ),
                "selected_context_mode": answer_eval.get("selected_context_mode"),
                "answer_prompt_style": answer_eval.get("answer_prompt_style"),
                "full_bad_output": answer_eval.get("full_bad_output"),
                "selected_bad_output": answer_eval.get("selected_bad_output"),
                "oracle_bad_output": answer_eval.get("oracle_bad_output"),
                "full_initial_bad_output": answer_eval.get("full_initial_bad_output"),
                "selected_initial_bad_output": answer_eval.get(
                    "selected_initial_bad_output"
                ),
                "oracle_initial_bad_output": answer_eval.get("oracle_initial_bad_output"),
                "full_generation_retried": answer_eval.get("full_generation_retried"),
                "selected_generation_retried": answer_eval.get(
                    "selected_generation_retried"
                ),
                "oracle_generation_retried": answer_eval.get("oracle_generation_retried"),
                "full_generation_used_retry": answer_eval.get("full_generation_used_retry"),
                "selected_generation_used_retry": answer_eval.get(
                    "selected_generation_used_retry"
                ),
                "oracle_generation_used_retry": answer_eval.get(
                    "oracle_generation_used_retry"
                ),
                "full_bad_output_reasons": answer_eval.get("full_bad_output_reasons", []),
                "selected_bad_output_reasons": answer_eval.get(
                    "selected_bad_output_reasons",
                    [],
                ),
                "oracle_bad_output_reasons": answer_eval.get("oracle_bad_output_reasons", []),
                "full_initial_bad_output_reasons": answer_eval.get(
                    "full_initial_bad_output_reasons",
                    [],
                ),
                "selected_initial_bad_output_reasons": answer_eval.get(
                    "selected_initial_bad_output_reasons",
                    [],
                ),
                "oracle_initial_bad_output_reasons": answer_eval.get(
                    "oracle_initial_bad_output_reasons",
                    [],
                ),
                "full_postprocess_actions": answer_eval.get("full_postprocess_actions", []),
                "selected_postprocess_actions": answer_eval.get(
                    "selected_postprocess_actions",
                    [],
                ),
                "oracle_postprocess_actions": answer_eval.get(
                    "oracle_postprocess_actions",
                    [],
                ),
                "bad_case_category": _build_bad_case_categories(result),
                "full_context_tokens": answer_eval.get("full_context_tokens"),
                "selected_context_tokens": answer_eval.get("selected_context_tokens"),
                "oracle_context_tokens": answer_eval.get("oracle_context_tokens"),
                "context_token_saving_ratio": answer_eval.get("context_token_saving_ratio"),
                "selected_unit_strategy": routing_eval.get("selected_unit_strategy"),
                "selected_topic_ids": qk_route_info.get("selected_topic_ids", []),
                "relevant_virtual_node_ids": result.get("relevant_virtual_nodes", []),
                "selected_virtual_node_ids": qk_route_info.get("selected_virtual_node_ids", []),
                "virtual_node_layout": route_info.get(
                    "virtual_node_layout",
                    qk_route_info.get("virtual_node_layout", []),
                ),
                "selected_turns": routing_eval.get("qk_selected_turns", []),
                "ordered_answer_turns": qk_route_info.get("ordered_answer_turns", []),
                "matched_turns": routing_eval.get("qk_matched_turns", []),
                "relevant_candidate_survival": qk_route_info.get(
                    "relevant_candidate_survival",
                    {},
                ),
                "query_budget_type": qk_route_info.get("query_budget_type"),
                "effective_route_top_k": qk_route_info.get("effective_route_top_k"),
                "route_selection_mode": qk_route_info.get("route_selection_mode"),
                "route_hybrid_core_ratio": qk_route_info.get("route_hybrid_core_ratio"),
                "route_hybrid_core_max_per_turn": qk_route_info.get(
                    "route_hybrid_core_max_per_turn"
                ),
                "route_selection_debug": qk_route_info.get("route_selection_debug", {}),
                "candidate_prefilter_mode": qk_route_info.get("candidate_prefilter_mode"),
                "candidate_prefilter_pool_size": qk_route_info.get(
                    "candidate_prefilter_pool_size"
                ),
                "candidate_prefilter_requested_pool_size": qk_route_info.get(
                    "candidate_prefilter_requested_pool_size"
                ),
                "candidate_prefilter_prune_ratio": qk_route_info.get(
                    "candidate_prefilter_prune_ratio"
                ),
                "candidate_prefilter_keep_ratio": qk_route_info.get(
                    "candidate_prefilter_keep_ratio"
                ),
                "candidate_prefilter_min_prune_ratio": qk_route_info.get(
                    "candidate_prefilter_min_prune_ratio"
                ),
                "candidate_prefilter_skip_reason": qk_route_info.get(
                    "candidate_prefilter_skip_reason",
                    "",
                ),
                "dynamic_candidate_pool_budget": qk_route_info.get(
                    "dynamic_candidate_pool_budget"
                ),
                "dynamic_candidate_pool_reason": qk_route_info.get(
                    "dynamic_candidate_pool_reason"
                ),
                "dynamic_candidate_pool_target": qk_route_info.get(
                    "dynamic_candidate_pool_target"
                ),
                "dynamic_candidate_pool_prune_ratio": qk_route_info.get(
                    "dynamic_candidate_pool_prune_ratio"
                ),
                "num_candidates_before_dynamic_pool": qk_route_info.get(
                    "num_candidates_before_dynamic_pool"
                ),
                "num_candidates_after_dynamic_pool": qk_route_info.get(
                    "num_candidates_after_dynamic_pool"
                ),
                "num_candidates_after_coarse_segment_gate": qk_route_info.get(
                    "num_candidates_after_coarse_segment_gate"
                ),
                "coarse_segment_gate_mode": qk_route_info.get("route_coarse_segment_gate"),
                "coarse_segment_gate_before": qk_route_info.get("coarse_segment_gate_before"),
                "coarse_segment_gate_after": qk_route_info.get("coarse_segment_gate_after"),
                "coarse_segment_gate_prune_ratio": qk_route_info.get(
                    "coarse_segment_gate_prune_ratio"
                ),
                "coarse_segment_gate_num_segments": qk_route_info.get(
                    "coarse_segment_gate_num_segments"
                ),
                "coarse_segment_gate_keep_segments": qk_route_info.get(
                    "coarse_segment_gate_keep_segments"
                ),
                "coarse_segment_gate_preview": qk_route_info.get("coarse_segment_gate_preview", []),
                "num_candidates_before_prefilter": qk_route_info.get(
                    "num_candidates_before_prefilter"
                ),
                "num_candidates_after_prefilter": qk_route_info.get(
                    "num_candidates_after_prefilter"
                ),
                "num_candidates_scored_qk": qk_route_info.get("num_candidates_scored_qk"),
                "answer_evidence_order": qk_route_info.get("answer_evidence_order"),
                "candidate_prefilter_preview": qk_route_info.get(
                    "candidate_prefilter_preview",
                    [],
                ),
                "dynamic_candidate_pool_preview": qk_route_info.get(
                    "dynamic_candidate_pool_preview",
                    [],
                ),
                "answer_turn_rerank_preview": qk_route_info.get(
                    "answer_turn_rerank_preview",
                    [],
                ),
                "transfer_accounting": result.get("transfer_accounting"),
                "virtual_node_transfer_accounting": result.get(
                    "virtual_node_transfer_accounting"
                ),
                "system_cost": result.get("system_cost"),
                "cachegen_full_estimate": result.get("cachegen_full_estimate"),
                "routing_timing": result.get("routing_timing"),
                "selected_chunks": result.get("selected_chunk_details", []),
                "selected_evidence_entries": result.get("selected_evidence_entries", []),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _preview_text(text, limit=260):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def write_case_answer_markdown(results, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# QMSum Answer Log\n\n")
        for result in results:
            answer_eval = result.get("answer_eval")
            if not answer_eval:
                continue
            qk_route_info = result.get("qk_route_info") or {}
            f.write(
                f"## doc {result.get('doc_id')} / query {result.get('query_idx')}\n\n"
            )
            f.write(f"- meeting_id: `{result.get('meeting_id', '')}`\n")
            f.write(f"- query: {result.get('query', '')}\n")
            f.write(
                f"- selected_topic_ids: "
                f"`{','.join(str(x) for x in qk_route_info.get('selected_topic_ids', []))}`\n"
            )
            f.write(
                f"- selected_virtual_node_ids: "
                f"`{','.join(str(x) for x in qk_route_info.get('selected_virtual_node_ids', []))}`\n"
            )
            f.write(
                f"- candidates prefilter/dyn/gate/qk: "
                f"`{qk_route_info.get('num_candidates_before_prefilter', '')}"
                f"->{qk_route_info.get('num_candidates_after_prefilter', '')}"
                f"->{qk_route_info.get('num_candidates_after_dynamic_pool', '')}"
                f"->{qk_route_info.get('num_candidates_after_coarse_segment_gate', '')}"
                f"->{qk_route_info.get('num_candidates_scored_qk', '')}`\n"
            )
            f.write(
                f"- F1 full/selected/oracle: "
                f"`{_as_float(answer_eval.get('full_answer_f1')):.3f}` / "
                f"`{_as_float(answer_eval.get('selected_answer_f1')):.3f}` / "
                f"`{_as_float(answer_eval.get('oracle_answer_f1')):.3f}`\n"
            )
            f.write(
                f"- context token saving: "
                f"`{100 * _as_float(answer_eval.get('context_token_saving_ratio')):.1f}%`\n\n"
            )
            f.write("**Gold**\n\n")
            f.write(_preview_text(answer_eval.get("gold_answer", ""), limit=800) + "\n\n")
            f.write("**Full**\n\n")
            f.write(_preview_text(answer_eval.get("full_answer", ""), limit=800) + "\n\n")
            f.write("**Selected**\n\n")
            f.write(_preview_text(answer_eval.get("selected_answer", ""), limit=800) + "\n\n")
            if answer_eval.get("oracle_answer"):
                f.write("**Oracle**\n\n")
                f.write(_preview_text(answer_eval.get("oracle_answer", ""), limit=800) + "\n\n")
