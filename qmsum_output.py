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
    dataset = getattr(args, "dataset", "qmsum")
    dataset_prefix = "" if dataset == "qmsum" else f"{dataset}_"
    if tag:
        return f"{dataset_prefix}N{args.num_nodes}_{args.start_doc}_{args.end_doc}_{tag}"
    return f"{dataset_prefix}N{args.num_nodes}_{args.start_doc}_{args.end_doc}"


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


def get_node_summary_gate_fields(result):
    qk_route_info = result.get("qk_route_info") or {}
    routing_eval = result.get("routing_eval") or {}
    return {
        "mode": qk_route_info.get(
            "node_summary_gate",
            routing_eval.get("route_node_summary_gate", "none"),
        ),
        "summary_mode": qk_route_info.get(
            "node_summary_gate_summary_mode",
            routing_eval.get("route_node_summary_gate_summary_mode", ""),
        ),
        "before_count": qk_route_info.get(
            "node_summary_gate_before",
            routing_eval.get("route_node_summary_gate_before", ""),
        ),
        "after_count": qk_route_info.get(
            "node_summary_gate_after",
            routing_eval.get("route_node_summary_gate_after", ""),
        ),
        "target_keep": qk_route_info.get(
            "node_summary_gate_target_keep",
            routing_eval.get("route_node_summary_gate_target_keep", ""),
        ),
        "prune_ratio": qk_route_info.get(
            "node_summary_gate_prune_ratio",
            routing_eval.get("route_node_summary_gate_prune_ratio", ""),
        ),
        "keep_ratio": qk_route_info.get("node_summary_gate_keep_ratio", ""),
        "budget_mode": qk_route_info.get("node_summary_gate_budget_mode", ""),
        "budget_reason": qk_route_info.get("node_summary_gate_budget_reason", ""),
        "adaptive_min_keep": qk_route_info.get(
            "node_summary_gate_adaptive_min_keep",
            "",
        ),
        "adaptive_gap": qk_route_info.get("node_summary_gate_adaptive_gap", ""),
        "adaptive_safety_factor": qk_route_info.get(
            "node_summary_gate_adaptive_safety_factor",
            "",
        ),
        "non_finite_score_count": qk_route_info.get(
            "node_summary_gate_non_finite_score_count",
            "",
        ),
        "score_pooling": qk_route_info.get(
            "node_summary_gate_score_pooling",
            "",
        ),
        "min_keep": qk_route_info.get("node_summary_gate_min_keep", ""),
        "max_keep": qk_route_info.get("node_summary_gate_max_keep", ""),
        "factor": qk_route_info.get("node_summary_gate_factor", ""),
        "populate_key_cache": qk_route_info.get(
            "node_summary_gate_populate_key_cache",
            "",
        ),
        "preview": qk_route_info.get("node_summary_gate_preview", []),
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


COMPACT_CASE_SUMMARY_COLUMNS = [
    "doc_id",
    "query_idx",
    "meeting_id",
    "selected_turn_hit",
    "selected_turn_recall",
    "selected_turn_precision",
    "selected_turn_f1",
    "survival_failure_stage",
    "survival_first_zero_stage",
    "survival_topic_filter_turn_recall",
    "survival_prefilter_turn_recall",
    "survival_dynamic_pool_turn_recall",
    "survival_coarse_gate_turn_recall",
    "survival_node_summary_gate_turn_recall",
    "survival_qk_scored_turn_recall",
    "survival_qk_selected_turn_recall",
    "qk_oracle_turn_recall_at_selected_budget",
    "qk_selector_gap",
    "turn_utility_recall_at_selected_budget",
    "query_budget_type",
    "effective_route_top_k",
    "candidate_topic_scope",
    "route_selection_mode",
    "candidate_prefilter_mode",
    "coarse_segment_gate_mode",
    "node_summary_gate_mode",
    "node_summary_gate_summary_mode",
    "node_summary_gate_budget_mode",
    "qk_aggregation",
    "qk_token_pooling",
    "num_candidates_before_prefilter",
    "num_candidates_after_prefilter",
    "num_candidates_after_coarse_segment_gate",
    "num_candidates_after_node_summary_gate",
    "num_candidates_scored_qk",
    "routing_overhead_ms",
    "routing_wall_clock_ms",
    "candidate_prefilter_ms",
    "dynamic_candidate_pool_ms",
    "coarse_segment_gate_ms",
    "node_summary_gate_ms",
    "qk_total_stage_ms",
    "selection_postprocess_ms",
    "selected_kv_mib",
    "full_kv_mib",
    "kv_saving_pct",
    "ttft_model",
    "query_dispatch_ms",
    "summary_metadata_fetch_ms",
    "node_local_selection_ms",
    "candidate_return_ms",
    "coordinator_fusion_ms",
    "offline_summary_prepare_ms",
    "oracle_exact_qk_ms_excluded",
    "legacy_selected_ttft_ms",
    "selected_ttft_ms",
    "full_ttft_ms",
    "ttft_saving_pct",
    "full_answer_f1",
    "selected_answer_f1",
    "answer_f1_delta",
    "ctx_token_saving_pct",
    "bad_case_category",
]


def _filter_case_summary_columns(header, row, mode):
    if mode == "full":
        return header, row
    if mode != "compact":
        raise ValueError(f"Unknown case summary mode: {mode}")

    keep_names = set(COMPACT_CASE_SUMMARY_COLUMNS)
    keep_indices = [idx for idx, name in enumerate(header) if name in keep_names]
    return [header[idx] for idx in keep_indices], [row[idx] for idx in keep_indices]


def write_case_summary_tsv(results, out_path, mode="compact"):
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
        "survival_node_summary_gate_turn_recall",
        "survival_qk_scored_turn_recall",
        "survival_qk_selected_turn_recall",
        "qk_oracle_turn_recall_at_selected_budget",
        "qk_selector_gap",
        "qk_budget_cap_gap",
        "qk_oracle_turn_hit_at_selected_budget",
        "qk_first_relevant_rank",
        "qk_first_relevant_rank_pct",
        "qk_relevant_turn_best_rank_mean",
        "qk_relevant_turn_best_rank_median",
        "qk_oracle_diagnosis",
        "qk_oracle_turns_at_selected_budget",
        "turn_utility_first_relevant_rank",
        "turn_utility_first_relevant_rank_pct",
        "turn_utility_mean_gold_rank",
        "turn_utility_recall_at_selected_budget",
        "turn_utility_gap_to_qk_oracle",
        "turn_utility_gap_to_scored",
        "turn_utility_ranked_turn_count",
        "turn_utility_top_budget_turns",
        "query_budget_type",
        "effective_route_top_k",
        "adaptive_topic_rescue",
        "adaptive_topic_rescue_triggered",
        "adaptive_topic_rescue_reason",
        "adaptive_topic_rescue_base_k",
        "adaptive_topic_rescue_final_k",
        "adaptive_topic_rescue_gap_ratio",
        "adaptive_topic_rescue_top1_ratio",
        "adaptive_topic_rescue_top1_score",
        "adaptive_topic_rescue_top2_score",
        "candidate_topic_scope",
        "route_selection_mode",
        "route_hybrid_core_ratio",
        "route_hybrid_core_max_per_turn",
        "route_pack_anchor_count",
        "route_pack_support_radius",
        "route_pack_max_turns",
        "route_pack_max_candidates",
        "topic_balanced_min_per_topic",
        "topic_balanced_topic_count",
        "topic_balanced_selected_topic_counts",
        "topic_soft_rescue_replacements",
        "topic_soft_rescue_selected_topic_counts",
        "pack_anchor_count",
        "pack_neighbor_support_added",
        "pack_same_turn_support_added",
        "pack_adjacent_turn_support_added",
        "pack_support_total_added",
        "pack_support_budget",
        "pack_coverage_budget",
        "pack_support_candidates_seen",
        "pack_support_candidates_below_threshold",
        "pack_replacement_budget",
        "pack_replacement_added",
        "pack_replacement_skipped_no_drop",
        "pack_replacement_skipped_low_score",
        "pack_replacement_score_tolerance",
        "pack_replacement_score_span",
        "pack_replacement_median_gap",
        "pack_coverage_turn_added",
        "pack_qk_backfill_added",
        "unique_turn_added",
        "global_unique_turn_added",
        "duplicate_deferred",
        "duplicate_backfill_added",
        "selection_selected_turn_count",
        "selection_duplicate_chunk_count",
        "candidate_prefilter_mode",
        "candidate_prefilter_pool_size",
        "candidate_prefilter_requested_pool_size",
        "candidate_prefilter_prune_ratio",
        "candidate_prefilter_keep_ratio",
        "candidate_prefilter_min_prune_ratio",
        "candidate_prefilter_skip_reason",
        "route_adaptive_prefilter",
        "candidate_prefilter_adaptive_keep_ratio",
        "candidate_prefilter_adaptive_uncertainty",
        "candidate_prefilter_adaptive_signal_count",
        "candidate_prefilter_adaptive_reason",
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
        "coarse_segment_gate_adaptive_enabled",
        "coarse_segment_gate_adaptive_keep_ratio",
        "coarse_segment_gate_adaptive_uncertainty",
        "coarse_segment_gate_adaptive_reason",
        "num_candidates_after_node_summary_gate",
        "node_summary_gate_mode",
        "node_summary_gate_summary_mode",
        "node_summary_gate_before",
        "node_summary_gate_after",
        "node_summary_gate_target_keep",
        "node_summary_gate_prune_ratio",
        "node_summary_gate_keep_ratio",
        "node_summary_gate_budget_mode",
        "node_summary_gate_budget_reason",
        "node_summary_gate_adaptive_min_keep",
        "node_summary_gate_adaptive_gap",
        "node_summary_gate_adaptive_safety_factor",
        "node_summary_gate_non_finite_score_count",
        "node_summary_gate_score_pooling",
        "node_summary_gate_min_keep",
        "node_summary_gate_max_keep",
        "node_summary_gate_factor",
        "node_summary_gate_populate_key_cache",
        "num_candidates_before_prefilter",
        "num_candidates_after_prefilter",
        "num_candidates_scored_qk",
        "qk_aggregation",
        "qk_topk",
        "qk_token_pooling",
        "qk_query_topk_ratio",
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
        "node_summary_gate_ms",
        "node_summary_prepare_ms",
        "node_summary_score_ms",
        "node_summary_aggregate_ms",
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
        "ttft_model",
        "query_dispatch_ms",
        "summary_metadata_fetch_ms",
        "node_local_selection_ms",
        "candidate_return_ms",
        "coordinator_fusion_ms",
        "offline_summary_prepare_ms",
        "oracle_exact_qk_ms_excluded",
        "legacy_selected_ttft_ms",
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
        "cachegen_roundtrip_status",
        "cachegen_roundtrip_answer_f1",
        "cachegen_roundtrip_delta_vs_full",
        "cachegen_roundtrip_delta_vs_selected",
        "selected_answer_f1_delta_vs_cachegen_roundtrip",
        "cachegen_roundtrip_bad_output",
        "cachegen_roundtrip_retried",
        "cachegen_roundtrip_used_retry",
        "cachegen_roundtrip_prompt_tokens",
        "cachegen_roundtrip_compressed_mib",
        "cachegen_roundtrip_compression_saving_pct",
        "cachegen_roundtrip_prefill_ms",
        "cachegen_roundtrip_encode_ms",
        "cachegen_roundtrip_decode_ms",
        "cachegen_roundtrip_generation_ms",
        "cachegen_roundtrip_fetch_latency_ms",
        "cachegen_roundtrip_ttft_ms",
        "selected_vs_cachegen_roundtrip_ttft_delta_ms",
        "selected_vs_cachegen_roundtrip_ttft_saving_pct",
        "cachegen_roundtrip_error_type",
        "cachegen_roundtrip_error",
        "full_answer_f1",
        "selected_answer_f1",
        "oracle_answer_available",
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
        output_header = None
        for result in results:
            routing_eval = result.get("routing_eval") or {}
            answer_eval = result.get("answer_eval") or {}
            qk_route_info = result.get("qk_route_info") or {}
            survival = qk_route_info.get("relevant_candidate_survival") or {}
            prefilter_fields = get_prefilter_fields(result)
            gate_fields = get_coarse_segment_gate_fields(result)
            node_gate_fields = get_node_summary_gate_fields(result)
            system_cost = result.get("system_cost") or {}
            routing_timing = result.get("routing_timing") or {}
            selected_cost = system_cost.get("selected", {})
            active_cost = system_cost.get("active_node_selected", {})
            legacy_cost = system_cost.get("legacy_selected", {})
            full_cost = system_cost.get("full", {})
            reduction = system_cost.get("reduction", {})
            ttft_assumptions = system_cost.get("assumptions", {})
            cachegen_full = result.get("cachegen_full_estimate") or {}
            cachegen_roundtrip = result.get("cachegen_roundtrip_answer") or {}
            bad_case_categories = _build_bad_case_categories(result)
            oracle_available = bool(answer_eval.get("oracle_answer_available", True))
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
                f"{_as_float(survival.get('node_summary_gate_turn_recall')):.4f}",
                f"{_as_float(survival.get('qk_scored_turn_recall')):.4f}",
                f"{_as_float(survival.get('qk_selected_turn_recall')):.4f}",
                f"{_as_float(survival.get('qk_oracle_turn_recall_at_selected_budget')):.4f}",
                f"{_as_float(survival.get('qk_selector_gap')):.4f}",
                f"{_as_float(survival.get('qk_budget_cap_gap')):.4f}",
                "1"
                if survival.get("qk_oracle_turn_hit_at_selected_budget", False)
                else "0",
                str(_as_int(survival.get("qk_first_relevant_rank"))),
                f"{_as_float(survival.get('qk_first_relevant_rank_pct')):.4f}",
                f"{_as_float(survival.get('qk_relevant_turn_best_rank_mean')):.2f}",
                f"{_as_float(survival.get('qk_relevant_turn_best_rank_median')):.2f}",
                str(survival.get("qk_oracle_diagnosis", "")),
                ",".join(
                    str(_as_int(x))
                    for x in survival.get(
                        "qk_oracle_turns_at_selected_budget",
                        [],
                    )
                ),
                str(_as_int(survival.get("turn_utility_first_relevant_rank"))),
                f"{_as_float(survival.get('turn_utility_first_relevant_rank_pct')):.4f}",
                f"{_as_float(survival.get('turn_utility_mean_gold_rank')):.2f}",
                f"{_as_float(survival.get('turn_utility_recall_at_selected_budget')):.4f}",
                f"{_as_float(survival.get('turn_utility_gap_to_qk_oracle')):.4f}",
                f"{_as_float(survival.get('turn_utility_gap_to_scored')):.4f}",
                str(_as_int(survival.get("turn_utility_ranked_turn_count"))),
                ",".join(
                    str(_as_int(x))
                    for x in survival.get("turn_utility_top_budget_turns", [])
                ),
                str(qk_route_info.get("query_budget_type", "")),
                str(qk_route_info.get("effective_route_top_k", "")),
                "1" if qk_route_info.get("adaptive_topic_rescue", False) else "0",
                "1" if qk_route_info.get("adaptive_topic_rescue_triggered", False) else "0",
                str(qk_route_info.get("adaptive_topic_rescue_reason", "")),
                str(qk_route_info.get("adaptive_topic_rescue_base_k", "")),
                str(qk_route_info.get("adaptive_topic_rescue_final_k", "")),
                f"{_as_float(qk_route_info.get('adaptive_topic_rescue_gap_ratio')):.4f}",
                f"{_as_float(qk_route_info.get('adaptive_topic_rescue_top1_ratio')):.4f}",
                f"{_as_float(qk_route_info.get('adaptive_topic_rescue_top1_score')):.4f}",
                f"{_as_float(qk_route_info.get('adaptive_topic_rescue_top2_score')):.4f}",
                str(qk_route_info.get("candidate_topic_scope", "")),
                str(qk_route_info.get("route_selection_mode", "chunk_topk")),
                str(qk_route_info.get("route_hybrid_core_ratio", "")),
                str(qk_route_info.get("route_hybrid_core_max_per_turn", "")),
                str(qk_route_info.get("route_pack_anchor_count", "")),
                str(qk_route_info.get("route_pack_support_radius", "")),
                str(qk_route_info.get("route_pack_max_turns", "")),
                str(qk_route_info.get("route_pack_max_candidates", "")),
                str(qk_route_info.get("topic_balanced_min_per_topic", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("topic_balanced_topic_count", "")),
                json.dumps(
                    (qk_route_info.get("route_selection_debug") or {}).get(
                        "topic_balanced_selected_topic_counts",
                        {},
                    ),
                    ensure_ascii=False,
                ),
                str((qk_route_info.get("route_selection_debug") or {}).get("topic_soft_rescue_replacements", "")),
                json.dumps(
                    (qk_route_info.get("route_selection_debug") or {}).get(
                        "topic_soft_rescue_selected_topic_counts",
                        {},
                    ),
                    ensure_ascii=False,
                ),
                str((qk_route_info.get("route_selection_debug") or {}).get("anchor_count", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("neighbor_support_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("same_turn_support_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("adjacent_turn_support_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("support_total_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("support_budget", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("coverage_budget", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("support_candidates_seen", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("support_candidates_below_threshold", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("replacement_budget", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("replacement_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("replacement_skipped_no_drop", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("replacement_skipped_low_score", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("replacement_score_tolerance", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("replacement_score_span", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("replacement_median_gap", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("coverage_turn_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("qk_backfill_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("unique_turn_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("global_unique_turn_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("duplicate_deferred", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("duplicate_backfill_added", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("selected_turn_count", "")),
                str((qk_route_info.get("route_selection_debug") or {}).get("selected_duplicate_chunk_count", "")),
                str(prefilter_fields.get("mode", "")),
                str(prefilter_fields.get("pool_size", "")),
                str(prefilter_fields.get("requested_pool_size", "")),
                str(prefilter_fields.get("prune_ratio", "")),
                str(prefilter_fields.get("keep_ratio", "")),
                str(prefilter_fields.get("min_prune_ratio", "")),
                str(prefilter_fields.get("skip_reason", "")),
                "1" if qk_route_info.get("route_adaptive_prefilter", False) else "0",
                f"{_as_float(qk_route_info.get('candidate_prefilter_adaptive_keep_ratio')):.4f}",
                f"{_as_float(qk_route_info.get('candidate_prefilter_adaptive_uncertainty')):.4f}",
                str(_as_int(qk_route_info.get("candidate_prefilter_adaptive_signal_count"))),
                str(qk_route_info.get("candidate_prefilter_adaptive_reason", "")),
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
                "1"
                if qk_route_info.get("coarse_segment_gate_adaptive_enabled", False)
                else "0",
                f"{_as_float(qk_route_info.get('coarse_segment_gate_adaptive_keep_ratio')):.4f}",
                f"{_as_float(qk_route_info.get('coarse_segment_gate_adaptive_uncertainty')):.4f}",
                str(qk_route_info.get("coarse_segment_gate_adaptive_reason", "")),
                str(qk_route_info.get("num_candidates_after_node_summary_gate", "")),
                str(node_gate_fields.get("mode", "")),
                str(node_gate_fields.get("summary_mode", "")),
                str(node_gate_fields.get("before_count", "")),
                str(node_gate_fields.get("after_count", "")),
                str(node_gate_fields.get("target_keep", "")),
                str(node_gate_fields.get("prune_ratio", "")),
                str(node_gate_fields.get("keep_ratio", "")),
                str(node_gate_fields.get("budget_mode", "")),
                str(node_gate_fields.get("budget_reason", "")),
                str(node_gate_fields.get("adaptive_min_keep", "")),
                str(node_gate_fields.get("adaptive_gap", "")),
                str(node_gate_fields.get("adaptive_safety_factor", "")),
                str(node_gate_fields.get("non_finite_score_count", "")),
                str(node_gate_fields.get("score_pooling", "")),
                str(node_gate_fields.get("min_keep", "")),
                str(node_gate_fields.get("max_keep", "")),
                str(node_gate_fields.get("factor", "")),
                "1" if node_gate_fields.get("populate_key_cache", False) else "0",
                str(prefilter_fields.get("before_count", "")),
                str(prefilter_fields.get("after_count", "")),
                str(qk_route_info.get("num_candidates_scored_qk", "")),
                str(qk_route_info.get("qk_aggregation", "")),
                str(qk_route_info.get("qk_topk", "")),
                str(qk_route_info.get("qk_token_pooling", "")),
                str(qk_route_info.get("qk_query_topk_ratio", "")),
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
                f"{_as_float(routing_timing.get('node_summary_gate_ms')):.2f}",
                f"{_as_float(routing_timing.get('node_summary_prepare_ms')):.2f}",
                f"{_as_float(routing_timing.get('node_summary_score_ms')):.2f}",
                f"{_as_float(routing_timing.get('node_summary_aggregate_ms')):.2f}",
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
                str(ttft_assumptions.get("timing_model", "")),
                f"{_as_float(active_cost.get('query_dispatch_ms')):.2f}",
                f"{_as_float(active_cost.get('summary_metadata_fetch_ms')):.2f}",
                f"{_as_float(active_cost.get('node_local_selection_ms')):.2f}",
                f"{_as_float(active_cost.get('candidate_return_ms')):.2f}",
                f"{_as_float(active_cost.get('coordinator_fusion_ms')):.2f}",
                f"{_as_float(active_cost.get('offline_summary_prepare_ms')):.2f}",
                f"{_as_float(active_cost.get('oracle_exact_qk_ms_excluded')):.2f}",
                f"{_as_float(legacy_cost.get('estimated_ttft_ms')):.2f}",
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
                str(cachegen_roundtrip.get("status", "")),
                f"{_as_float(cachegen_roundtrip.get('answer_f1')):.4f}",
                f"{_as_float(cachegen_roundtrip.get('answer_f1_delta_vs_full')):.4f}",
                f"{_as_float(cachegen_roundtrip.get('answer_f1_delta_vs_selected')):.4f}",
                f"{_as_float(cachegen_roundtrip.get('selected_answer_f1_delta_vs_cachegen_roundtrip')):.4f}",
                "1" if cachegen_roundtrip.get("bad_output", False) else "0",
                "1" if cachegen_roundtrip.get("retried", False) else "0",
                "1" if cachegen_roundtrip.get("used_retry", False) else "0",
                str(_as_int(cachegen_roundtrip.get("prompt_tokens"))),
                f"{_as_float(cachegen_roundtrip.get('compressed_mib')):.2f}",
                f"{100.0 * _as_float(cachegen_roundtrip.get('compression_saving_ratio')):.1f}",
                f"{_as_float(cachegen_roundtrip.get('prefill_ms')):.2f}",
                f"{_as_float(cachegen_roundtrip.get('total_encode_ms')):.2f}",
                f"{_as_float(cachegen_roundtrip.get('measured_decode_ms')):.2f}",
                f"{_as_float(cachegen_roundtrip.get('generation_ms')):.2f}",
                f"{_as_float(cachegen_roundtrip.get('fetch_latency_ms')):.2f}",
                f"{_as_float(cachegen_roundtrip.get('estimated_ttft_ms')):.2f}",
                f"{_as_float(cachegen_roundtrip.get('selected_vs_cachegen_roundtrip_ttft_delta_ms')):.2f}",
                f"{100.0 * _as_float(cachegen_roundtrip.get('selected_vs_cachegen_roundtrip_ttft_saving_ratio')):.1f}",
                str(cachegen_roundtrip.get("error_type", "")),
                str(cachegen_roundtrip.get("error", cachegen_roundtrip.get("skip_reason", ""))),
                f"{_as_float(answer_eval.get('full_answer_f1')):.4f}",
                f"{_as_float(answer_eval.get('selected_answer_f1')):.4f}",
                "1" if oracle_available else "0",
                f"{_as_float(answer_eval.get('oracle_answer_f1')):.4f}"
                if oracle_available
                else "",
                f"{_as_float(answer_eval.get('answer_f1_delta')):.4f}",
                f"{_as_float(answer_eval.get('oracle_answer_f1_delta_vs_full')):.4f}"
                if oracle_available
                else "",
                f"{_as_float(answer_eval.get('selected_answer_f1_delta_vs_oracle')):.4f}"
                if oracle_available
                else "",
                "1" if answer_eval.get("full_bad_output", False) else "0",
                "1" if answer_eval.get("selected_bad_output", False) else "0",
                "1" if oracle_available and answer_eval.get("oracle_bad_output", False) else "0",
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
            filtered_header, filtered_row = _filter_case_summary_columns(
                header,
                row,
                mode,
            )
            if output_header is None:
                output_header = filtered_header
                f.write("\t".join(output_header) + "\n")
            f.write("\t".join(filtered_row) + "\n")
        if output_header is None:
            filtered_header, _ = _filter_case_summary_columns(header, header, mode)
            f.write("\t".join(filtered_header) + "\n")


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
                "oracle_answer_available": answer_eval.get(
                    "oracle_answer_available",
                    True,
                ),
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
                "coarse_selected_topic_ids": qk_route_info.get(
                    "coarse_selected_topic_ids",
                    [],
                ),
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
                "qk_oracle_diagnosis": (
                    qk_route_info.get("relevant_candidate_survival", {}) or {}
                ).get("qk_oracle_diagnosis"),
                "qk_oracle_turn_recall_at_selected_budget": (
                    qk_route_info.get("relevant_candidate_survival", {}) or {}
                ).get("qk_oracle_turn_recall_at_selected_budget"),
                "qk_selector_gap": (
                    qk_route_info.get("relevant_candidate_survival", {}) or {}
                ).get("qk_selector_gap"),
                "qk_first_relevant_rank": (
                    qk_route_info.get("relevant_candidate_survival", {}) or {}
                ).get("qk_first_relevant_rank"),
                "turn_utility_first_relevant_rank": (
                    qk_route_info.get("relevant_candidate_survival", {}) or {}
                ).get("turn_utility_first_relevant_rank"),
                "turn_utility_recall_at_selected_budget": (
                    qk_route_info.get("relevant_candidate_survival", {}) or {}
                ).get("turn_utility_recall_at_selected_budget"),
                "turn_utility_gap_to_qk_oracle": (
                    qk_route_info.get("relevant_candidate_survival", {}) or {}
                ).get("turn_utility_gap_to_qk_oracle"),
                "turn_utility_gold_preview": (
                    qk_route_info.get("relevant_candidate_survival", {}) or {}
                ).get("turn_utility_gold_preview", []),
                "turn_utility_top_budget_turns": (
                    qk_route_info.get("relevant_candidate_survival", {}) or {}
                ).get("turn_utility_top_budget_turns", []),
                "query_budget_type": qk_route_info.get("query_budget_type"),
                "effective_route_top_k": qk_route_info.get("effective_route_top_k"),
                "adaptive_topic_rescue": qk_route_info.get("adaptive_topic_rescue"),
                "adaptive_topic_rescue_triggered": qk_route_info.get(
                    "adaptive_topic_rescue_triggered"
                ),
                "adaptive_topic_rescue_reason": qk_route_info.get(
                    "adaptive_topic_rescue_reason"
                ),
                "adaptive_topic_rescue_base_k": qk_route_info.get(
                    "adaptive_topic_rescue_base_k"
                ),
                "adaptive_topic_rescue_final_k": qk_route_info.get(
                    "adaptive_topic_rescue_final_k"
                ),
                "adaptive_topic_rescue_gap_ratio": qk_route_info.get(
                    "adaptive_topic_rescue_gap_ratio"
                ),
                "adaptive_topic_rescue_top1_ratio": qk_route_info.get(
                    "adaptive_topic_rescue_top1_ratio"
                ),
                "adaptive_topic_rescue_top1_score": qk_route_info.get(
                    "adaptive_topic_rescue_top1_score"
                ),
                "adaptive_topic_rescue_top2_score": qk_route_info.get(
                    "adaptive_topic_rescue_top2_score"
                ),
                "candidate_topic_scope": qk_route_info.get("candidate_topic_scope"),
                "route_selection_mode": qk_route_info.get("route_selection_mode"),
                "route_hybrid_core_ratio": qk_route_info.get("route_hybrid_core_ratio"),
                "route_hybrid_core_max_per_turn": qk_route_info.get(
                    "route_hybrid_core_max_per_turn"
                ),
                "route_pack_anchor_count": qk_route_info.get("route_pack_anchor_count"),
                "route_pack_support_radius": qk_route_info.get(
                    "route_pack_support_radius"
                ),
                "route_pack_max_turns": qk_route_info.get("route_pack_max_turns"),
                "route_pack_max_candidates": qk_route_info.get(
                    "route_pack_max_candidates"
                ),
                "route_pack_support_score_ratio": qk_route_info.get(
                    "route_pack_support_score_ratio"
                ),
                "route_pack_support_same_turn": qk_route_info.get(
                    "route_pack_support_same_turn"
                ),
                "topic_balanced_min_per_topic": qk_route_info.get(
                    "topic_balanced_min_per_topic"
                ),
                "topic_soft_rescue_max_replacements": qk_route_info.get(
                    "topic_soft_rescue_max_replacements"
                ),
                "topic_soft_rescue_margin_ratio": qk_route_info.get(
                    "topic_soft_rescue_margin_ratio"
                ),
                "topic_soft_rescue_min_score_ratio": qk_route_info.get(
                    "topic_soft_rescue_min_score_ratio"
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
                "route_adaptive_prefilter": qk_route_info.get(
                    "route_adaptive_prefilter"
                ),
                "route_adaptive_coarse_segment_gate": qk_route_info.get(
                    "route_adaptive_coarse_segment_gate"
                ),
                "route_adaptive_min_keep_ratio": qk_route_info.get(
                    "route_adaptive_min_keep_ratio"
                ),
                "route_adaptive_max_keep_ratio": qk_route_info.get(
                    "route_adaptive_max_keep_ratio"
                ),
                "candidate_prefilter_adaptive_keep_ratio": qk_route_info.get(
                    "candidate_prefilter_adaptive_keep_ratio"
                ),
                "candidate_prefilter_adaptive_uncertainty": qk_route_info.get(
                    "candidate_prefilter_adaptive_uncertainty"
                ),
                "candidate_prefilter_adaptive_signal_count": qk_route_info.get(
                    "candidate_prefilter_adaptive_signal_count"
                ),
                "candidate_prefilter_adaptive_score_span": qk_route_info.get(
                    "candidate_prefilter_adaptive_score_span"
                ),
                "candidate_prefilter_adaptive_reason": qk_route_info.get(
                    "candidate_prefilter_adaptive_reason"
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
                "coarse_segment_gate_adaptive_enabled": qk_route_info.get(
                    "coarse_segment_gate_adaptive_enabled"
                ),
                "coarse_segment_gate_adaptive_keep_ratio": qk_route_info.get(
                    "coarse_segment_gate_adaptive_keep_ratio"
                ),
                "coarse_segment_gate_adaptive_uncertainty": qk_route_info.get(
                    "coarse_segment_gate_adaptive_uncertainty"
                ),
                "coarse_segment_gate_adaptive_reason": qk_route_info.get(
                    "coarse_segment_gate_adaptive_reason"
                ),
                "coarse_segment_gate_num_segments": qk_route_info.get(
                    "coarse_segment_gate_num_segments"
                ),
                "coarse_segment_gate_keep_segments": qk_route_info.get(
                    "coarse_segment_gate_keep_segments"
                ),
                "coarse_segment_gate_preview": qk_route_info.get("coarse_segment_gate_preview", []),
                "node_summary_gate_mode": qk_route_info.get("node_summary_gate"),
                "node_summary_gate_summary_mode": qk_route_info.get(
                    "node_summary_gate_summary_mode"
                ),
                "node_summary_gate_before": qk_route_info.get("node_summary_gate_before"),
                "node_summary_gate_after": qk_route_info.get("node_summary_gate_after"),
                "node_summary_gate_target_keep": qk_route_info.get(
                    "node_summary_gate_target_keep"
                ),
                "node_summary_gate_prune_ratio": qk_route_info.get(
                    "node_summary_gate_prune_ratio"
                ),
                "node_summary_gate_keep_ratio": qk_route_info.get(
                    "node_summary_gate_keep_ratio"
                ),
                "node_summary_gate_budget_mode": qk_route_info.get(
                    "node_summary_gate_budget_mode"
                ),
                "node_summary_gate_budget_reason": qk_route_info.get(
                    "node_summary_gate_budget_reason"
                ),
                "node_summary_gate_adaptive_min_keep": qk_route_info.get(
                    "node_summary_gate_adaptive_min_keep"
                ),
                "node_summary_gate_adaptive_gap": qk_route_info.get(
                    "node_summary_gate_adaptive_gap"
                ),
                "node_summary_gate_adaptive_safety_factor": qk_route_info.get(
                    "node_summary_gate_adaptive_safety_factor"
                ),
                "node_summary_gate_non_finite_score_count": qk_route_info.get(
                    "node_summary_gate_non_finite_score_count"
                ),
                "node_summary_gate_score_pooling": qk_route_info.get(
                    "node_summary_gate_score_pooling"
                ),
                "node_summary_gate_min_keep": qk_route_info.get(
                    "node_summary_gate_min_keep"
                ),
                "node_summary_gate_max_keep": qk_route_info.get(
                    "node_summary_gate_max_keep"
                ),
                "node_summary_gate_factor": qk_route_info.get("node_summary_gate_factor"),
                "node_summary_gate_populate_key_cache": qk_route_info.get(
                    "node_summary_gate_populate_key_cache"
                ),
                "num_candidates_after_node_summary_gate": qk_route_info.get(
                    "num_candidates_after_node_summary_gate"
                ),
                "node_summary_gate_preview": qk_route_info.get(
                    "node_summary_gate_preview",
                    [],
                ),
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
                "cachegen_roundtrip_answer": result.get("cachegen_roundtrip_answer"),
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
                f"- candidates prefilter/dyn/coarse/node-summary/qk: "
                f"`{qk_route_info.get('num_candidates_before_prefilter', '')}"
                f"->{qk_route_info.get('num_candidates_after_prefilter', '')}"
                f"->{qk_route_info.get('num_candidates_after_dynamic_pool', '')}"
                f"->{qk_route_info.get('num_candidates_after_coarse_segment_gate', '')}"
                f"->{qk_route_info.get('num_candidates_after_node_summary_gate', '')}"
                f"->{qk_route_info.get('num_candidates_scored_qk', '')}`\n"
            )
            if str(qk_route_info.get("node_summary_gate", "none")) != "none":
                f.write(
                    f"- node_summary_gate: "
                    f"`{qk_route_info.get('node_summary_gate', '')}` / "
                    f"`{qk_route_info.get('node_summary_gate_summary_mode', '')}` "
                    f"target `{qk_route_info.get('node_summary_gate_target_keep', '')}`, "
                    f"prune `{100 * _as_float(qk_route_info.get('node_summary_gate_prune_ratio')):.1f}%`\n"
                )
            f.write(
                f"- F1 full/selected/oracle: "
                f"`{_as_float(answer_eval.get('full_answer_f1')):.3f}` / "
                f"`{_as_float(answer_eval.get('selected_answer_f1')):.3f}` / "
            )
            if answer_eval.get("oracle_answer_available", True):
                f.write(f"`{_as_float(answer_eval.get('oracle_answer_f1')):.3f}`\n")
            else:
                f.write("`skipped`\n")
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
            cachegen_roundtrip = result.get("cachegen_roundtrip_answer") or {}
            if cachegen_roundtrip:
                f.write("**CacheGen Roundtrip**\n\n")
                if cachegen_roundtrip.get("status") == "ok":
                    f.write(
                        f"F1 `{_as_float(cachegen_roundtrip.get('answer_f1')):.3f}`, "
                        f"TTFT `{_as_float(cachegen_roundtrip.get('estimated_ttft_ms')):.2f} ms`, "
                        f"decode `{_as_float(cachegen_roundtrip.get('measured_decode_ms')):.2f} ms`\n\n"
                    )
                    f.write(_preview_text(cachegen_roundtrip.get("answer", ""), limit=800) + "\n\n")
                else:
                    f.write(
                        f"`{cachegen_roundtrip.get('status', 'error')}` "
                        f"{cachegen_roundtrip.get('error_type', '')}: "
                        f"{cachegen_roundtrip.get('error', cachegen_roundtrip.get('skip_reason', ''))}\n\n"
                    )
