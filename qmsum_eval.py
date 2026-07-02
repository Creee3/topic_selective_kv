from collections import defaultdict

import numpy as np


def compute_selected_turn_metrics(selected_candidates, relevant_turns):
    selected_turns = sorted({int(c["turn_idx"]) for c in selected_candidates})
    relevant_turns = set(int(x) for x in relevant_turns)
    matched_turns = sorted(set(selected_turns) & relevant_turns)
    recall = len(matched_turns) / len(relevant_turns) if relevant_turns else 0.0
    precision = len(matched_turns) / len(selected_turns) if selected_turns else 0.0
    f1 = 0.0
    if recall + precision > 0:
        f1 = 2.0 * recall * precision / (recall + precision)
    return {
        "selected_turns": selected_turns,
        "matched_turns": matched_turns,
        "hit": bool(matched_turns),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
    }


def coalesce_selected_candidates(selected_candidates, unit_field=None):
    ordered = sorted(
        selected_candidates,
        key=lambda c: (int(c["start_t"]), int(c["end_t"]), int(c["turn_idx"])),
    )
    segments = []
    for cand in ordered:
        start_t = int(cand["start_t"])
        end_t = int(cand["end_t"])
        turn_idx = int(cand["turn_idx"])
        unit_id = cand.get(unit_field) if unit_field else None
        unit_ids = [int(unit_id)] if unit_id is not None else []

        if not segments or start_t > segments[-1]["end_t"]:
            segments.append(
                {
                    "start_t": start_t,
                    "end_t": end_t,
                    "n_tokens": max(0, end_t - start_t),
                    "num_chunks": 1,
                    "turn_start": turn_idx,
                    "turn_end": turn_idx,
                    "unit_ids": unit_ids,
                }
            )
            continue

        seg = segments[-1]
        seg["end_t"] = max(seg["end_t"], end_t)
        seg["n_tokens"] = max(0, seg["end_t"] - seg["start_t"])
        seg["num_chunks"] += 1
        seg["turn_start"] = min(seg["turn_start"], turn_idx)
        seg["turn_end"] = max(seg["turn_end"], turn_idx)
        if unit_id is not None and int(unit_id) not in seg["unit_ids"]:
            seg["unit_ids"].append(int(unit_id))

    for seg in segments:
        seg["unit_ids"] = sorted(seg["unit_ids"])
    return segments


def build_transfer_accounting(selected_candidates, transfer_unit_type, unit_field):
    selected_chunk_count = len(selected_candidates)
    selected_token_count = int(sum(int(c.get("n_tokens", 0)) for c in selected_candidates))
    if selected_chunk_count == 0:
        return {
            "transfer_unit_type": transfer_unit_type,
            "selected_chunk_count": 0,
            "selected_token_count": 0,
            "unique_transfer_unit_count": 0,
            "unique_transfer_units": [],
            "global_contiguous_segment_count": 0,
            "transfer_segment_count": 0,
            "avg_chunks_per_global_segment": 0.0,
            "avg_chunks_per_transfer_segment": 0.0,
            "coalescing_gain_global": 0.0,
            "coalescing_gain_transfer": 0.0,
            "per_unit_chunk_counts": {},
            "per_unit_token_counts": {},
            "per_unit_segment_counts": {},
            "global_segments_preview": [],
            "transfer_cost_proxy": {"selected_tokens": 0, "transfer_segments": 0},
        }

    global_segments = coalesce_selected_candidates(selected_candidates)
    by_unit = defaultdict(list)
    per_unit_chunk_counts = defaultdict(int)
    per_unit_token_counts = defaultdict(int)
    for cand in selected_candidates:
        unit_id = int(cand.get(unit_field, -1))
        by_unit[unit_id].append(cand)
        per_unit_chunk_counts[unit_id] += 1
        per_unit_token_counts[unit_id] += int(cand.get("n_tokens", 0))

    unique_transfer_units = sorted(by_unit.keys())
    per_unit_segment_counts = {}
    transfer_segment_count = 0
    for unit_id, cands in by_unit.items():
        unit_segments = coalesce_selected_candidates(cands)
        per_unit_segment_counts[unit_id] = len(unit_segments)
        transfer_segment_count += len(unit_segments)

    global_segment_count = len(global_segments)
    avg_chunks_per_global_segment = (
        selected_chunk_count / global_segment_count if global_segment_count else 0.0
    )
    avg_chunks_per_transfer_segment = (
        selected_chunk_count / transfer_segment_count if transfer_segment_count else 0.0
    )
    coalescing_gain_global = 1.0 - global_segment_count / selected_chunk_count
    coalescing_gain_transfer = 1.0 - transfer_segment_count / selected_chunk_count

    return {
        "transfer_unit_type": transfer_unit_type,
        "selected_chunk_count": int(selected_chunk_count),
        "selected_token_count": int(selected_token_count),
        "unique_transfer_unit_count": int(len(unique_transfer_units)),
        "unique_transfer_units": [int(x) for x in unique_transfer_units],
        "global_contiguous_segment_count": int(global_segment_count),
        "transfer_segment_count": int(transfer_segment_count),
        "avg_chunks_per_global_segment": float(avg_chunks_per_global_segment),
        "avg_chunks_per_transfer_segment": float(avg_chunks_per_transfer_segment),
        "coalescing_gain_global": float(coalescing_gain_global),
        "coalescing_gain_transfer": float(coalescing_gain_transfer),
        "per_unit_chunk_counts": {
            str(k): int(v) for k, v in sorted(per_unit_chunk_counts.items())
        },
        "per_unit_token_counts": {
            str(k): int(v) for k, v in sorted(per_unit_token_counts.items())
        },
        "per_unit_segment_counts": {
            str(k): int(v) for k, v in sorted(per_unit_segment_counts.items())
        },
        "global_segments_preview": global_segments[:20],
        "transfer_cost_proxy": {
            "selected_tokens": int(selected_token_count),
            "transfer_segments": int(transfer_segment_count),
        },
    }


def build_system_cost_config(model_config, args):
    num_layers = int(getattr(model_config, "num_hidden_layers", 0))
    num_attention_heads = int(getattr(model_config, "num_attention_heads", 0))
    num_key_value_heads = int(
        getattr(model_config, "num_key_value_heads", num_attention_heads)
    )
    head_dim = int(
        getattr(
            model_config,
            "head_dim",
            getattr(model_config, "hidden_size", 0) // max(1, num_attention_heads),
        )
    )
    kv_cache_dtype_bytes = float(getattr(args, "kv_cache_dtype_bytes", 2.0))
    kv_bytes_per_token = 2.0 * num_layers * num_key_value_heads * head_dim * kv_cache_dtype_bytes
    bandwidth_gbps = float(getattr(args, "fetch_bandwidth_gbps", 25.0))
    bandwidth_bytes_per_ms = bandwidth_gbps * 1_000_000_000.0 / 8.0 / 1000.0
    control_bandwidth_gbps = float(getattr(args, "control_bandwidth_gbps", bandwidth_gbps))
    control_bandwidth_bytes_per_ms = (
        control_bandwidth_gbps * 1_000_000_000.0 / 8.0 / 1000.0
    )
    return {
        "ttft_model": str(getattr(args, "ttft_model", "active_node_v2")),
        "num_hidden_layers": int(num_layers),
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim),
        "kv_cache_dtype_bytes": float(kv_cache_dtype_bytes),
        "kv_bytes_per_token": float(kv_bytes_per_token),
        "fetch_bandwidth_gbps": float(bandwidth_gbps),
        "fetch_bandwidth_bytes_per_ms": float(bandwidth_bytes_per_ms),
        "control_bandwidth_gbps": float(control_bandwidth_gbps),
        "control_bandwidth_bytes_per_ms": float(control_bandwidth_bytes_per_ms),
        "per_node_rtt_ms": float(getattr(args, "per_node_rtt_ms", 1.0)),
        "control_rtt_ms": float(
            getattr(args, "control_rtt_ms", getattr(args, "per_node_rtt_ms", 1.0))
        ),
        "per_segment_overhead_ms": float(
            getattr(args, "per_segment_overhead_ms", 0.15)
        ),
        "per_rpc_overhead_ms": float(getattr(args, "per_rpc_overhead_ms", 0.05)),
        "decode_startup_ms": float(getattr(args, "decode_startup_ms", 15.0)),
        "candidate_metadata_bytes": float(getattr(args, "candidate_metadata_bytes", 64.0)),
        "query_request_bytes": float(getattr(args, "query_request_bytes", 4096.0)),
    }


def _summary_prototypes_per_block(summary_mode):
    mode = str(summary_mode or "mean_key").lower()
    if mode in {"multi_key", "mean_peak_boundary"}:
        return 3
    if mode in {"quest_minmax", "minmax_key"}:
        return 2
    return 1


def _estimate_summary_metadata_bytes(candidate_count, summary_mode, cost_config):
    prototypes = _summary_prototypes_per_block(summary_mode)
    return (
        float(max(0, int(candidate_count)))
        * float(cost_config.get("num_key_value_heads", 0))
        * float(cost_config.get("head_dim", 0))
        * float(cost_config.get("kv_cache_dtype_bytes", 0.0))
        * float(prototypes)
    )


def _estimate_transfer_case(
    token_count,
    transfer_unit_count,
    transfer_segment_count,
    compute_time_ms,
    cost_config,
    parallel_fetch=False,
    node_token_counts=None,
    node_segment_counts=None,
):
    kv_bytes_per_token = float(cost_config.get("kv_bytes_per_token", 0.0))
    bandwidth_bytes_per_ms = max(
        1e-9,
        float(cost_config.get("fetch_bandwidth_bytes_per_ms", 1.0)),
    )
    per_node_rtt_ms = float(cost_config.get("per_node_rtt_ms", 0.0))
    per_segment_overhead_ms = float(cost_config.get("per_segment_overhead_ms", 0.0))
    decode_startup_ms = float(cost_config.get("decode_startup_ms", 0.0))

    kv_bytes = float(token_count) * kv_bytes_per_token
    bandwidth_time_ms = kv_bytes / bandwidth_bytes_per_ms
    node_rtt_time_ms = float(transfer_unit_count) * per_node_rtt_ms
    segment_time_ms = float(transfer_segment_count) * per_segment_overhead_ms
    serial_transfer_time_ms = bandwidth_time_ms + node_rtt_time_ms + segment_time_ms

    parallel_critical_path_ms = serial_transfer_time_ms
    if parallel_fetch:
        node_token_counts = node_token_counts or {}
        node_segment_counts = node_segment_counts or {}
        node_times = []
        for node_id, node_tokens in node_token_counts.items():
            node_segments = int(node_segment_counts.get(node_id, 1))
            node_bytes = float(node_tokens) * kv_bytes_per_token
            node_times.append(
                node_bytes / bandwidth_bytes_per_ms
                + per_node_rtt_ms
                + float(node_segments) * per_segment_overhead_ms
            )
        if node_times:
            parallel_critical_path_ms = max(bandwidth_time_ms, max(node_times))

    transfer_time_ms = parallel_critical_path_ms if parallel_fetch else serial_transfer_time_ms
    decode_time_ms = decode_startup_ms
    estimated_ttft_ms = float(compute_time_ms) + transfer_time_ms + decode_time_ms
    return {
        "token_count": int(token_count),
        "transfer_unit_count": int(transfer_unit_count),
        "transfer_segment_count": int(transfer_segment_count),
        "compute_time_ms": float(compute_time_ms),
        "kv_bytes": float(kv_bytes),
        "kv_mib": float(kv_bytes / (1024.0 * 1024.0)),
        "bandwidth_time_ms": float(bandwidth_time_ms),
        "node_rtt_time_ms": float(node_rtt_time_ms),
        "segment_overhead_time_ms": float(segment_time_ms),
        "serial_transfer_time_ms": float(serial_transfer_time_ms),
        "parallel_transfer_time_ms": float(parallel_critical_path_ms),
        "transfer_time_ms": float(transfer_time_ms),
        "fetch_latency_ms": float(transfer_time_ms),
        "decode_time_ms": float(decode_time_ms),
        "component_sum_check_ms": float(estimated_ttft_ms),
        "estimated_ttft_ms": float(estimated_ttft_ms),
    }


def build_system_cost_estimate(
    total_tokens,
    num_virtual_nodes,
    transfer_accounting,
    routing_overhead_ms,
    cost_config,
    routing_wall_clock_ms=None,
    simulator_excluded_ms=0.0,
    routing_breakdown=None,
    qk_route_info=None,
    selected_candidates=None,
):
    routing_breakdown = routing_breakdown or {}
    qk_route_info = qk_route_info or {}
    selected_candidates = list(selected_candidates or [])
    ttft_model = str(cost_config.get("ttft_model", "active_node_v2")).lower()
    per_node_rtt_ms = float(cost_config.get("per_node_rtt_ms", 0.0))
    control_rtt_ms = float(cost_config.get("control_rtt_ms", per_node_rtt_ms))
    per_segment_overhead_ms = float(cost_config.get("per_segment_overhead_ms", 0.0))
    per_rpc_overhead_ms = float(cost_config.get("per_rpc_overhead_ms", 0.0))
    decode_startup_ms = float(cost_config.get("decode_startup_ms", 0.0))
    control_bandwidth_bytes_per_ms = max(
        1e-9,
        float(cost_config.get("control_bandwidth_bytes_per_ms", 1.0)),
    )
    candidate_metadata_bytes = float(cost_config.get("candidate_metadata_bytes", 64.0))
    query_request_bytes = float(cost_config.get("query_request_bytes", 4096.0))

    selected_node_token_counts = defaultdict(int)
    selected_node_segment_counts = defaultdict(int)
    for cand in selected_candidates:
        node_id = int(cand.get("transfer_node_id", -1))
        if node_id < 0:
            continue
        selected_node_token_counts[node_id] += int(cand.get("n_tokens", 0))
        selected_node_segment_counts[node_id] += 1

    legacy_selected = _estimate_transfer_case(
        transfer_accounting.get("selected_token_count", 0),
        transfer_accounting.get("unique_transfer_unit_count", 0),
        transfer_accounting.get("transfer_segment_count", 0),
        routing_overhead_ms,
        cost_config,
    )
    full_transfer_unit_count = max(0, int(num_virtual_nodes))

    contacted_nodes = int(
        transfer_accounting.get("unique_transfer_unit_count", 0)
        or len(qk_route_info.get("selected_virtual_node_ids", []) or [])
        or full_transfer_unit_count
    )
    contacted_nodes = max(0, contacted_nodes)
    selected_segments = int(transfer_accounting.get("transfer_segment_count", 0))
    selected_candidate_count = int(transfer_accounting.get("selected_chunk_count", 0))
    summary_candidate_count = int(
        qk_route_info.get(
            "node_summary_gate_before",
            qk_route_info.get(
                "num_candidates_after_coarse_segment_gate",
                qk_route_info.get("num_candidates_scored_qk", selected_candidate_count),
            ),
        )
        or 0
    )
    returned_candidate_count = int(
        qk_route_info.get("node_summary_gate_after", selected_candidate_count) or 0
    )
    summary_mode = qk_route_info.get("node_summary_gate_summary_mode", "mean_key")
    summary_bytes = _estimate_summary_metadata_bytes(
        summary_candidate_count,
        summary_mode,
        cost_config,
    )
    query_dispatch_ms = (
        query_request_bytes / control_bandwidth_bytes_per_ms
        + float(contacted_nodes) * (control_rtt_ms + per_rpc_overhead_ms)
    )
    summary_metadata_fetch_ms = summary_bytes / control_bandwidth_bytes_per_ms
    candidate_return_bytes = float(returned_candidate_count) * candidate_metadata_bytes
    candidate_return_ms = (
        candidate_return_bytes / control_bandwidth_bytes_per_ms
        + float(contacted_nodes) * (control_rtt_ms + per_rpc_overhead_ms)
    )

    node_summary_score_ms = float(routing_breakdown.get("node_summary_score_ms", 0.0))
    node_summary_aggregate_ms = float(routing_breakdown.get("node_summary_aggregate_ms", 0.0))
    query_q_prepare_ms = float(routing_breakdown.get("query_q_prepare_ms", 0.0))
    # Query-Q preparation is shared request-side work: the selected path uses it
    # for summary/Q-K routing, while a full-KV baseline still needs equivalent
    # query prefill before first-token decode. Charge it to both for active v2.
    shared_query_compute_ms = query_q_prepare_ms if ttft_model == "active_node_v2" else 0.0
    selection_postprocess_ms = float(routing_breakdown.get("selection_postprocess_ms", 0.0))
    coarse_routing_ms = (
        float(routing_breakdown.get("coarse_topic_routing_ms", 0.0))
        + float(routing_breakdown.get("topic_filter_ms", 0.0))
        + float(routing_breakdown.get("candidate_prefilter_ms", 0.0))
        + float(routing_breakdown.get("dynamic_candidate_pool_ms", 0.0))
        + float(routing_breakdown.get("coarse_segment_gate_ms", 0.0))
    )
    node_local_selection_ms = (
        node_summary_score_ms / max(1, contacted_nodes)
        if contacted_nodes > 1
        else node_summary_score_ms
    )
    coordinator_fusion_ms = node_summary_aggregate_ms + selection_postprocess_ms
    active_compute_time_ms = (
        query_q_prepare_ms
        + coarse_routing_ms
        + query_dispatch_ms
        + summary_metadata_fetch_ms
        + node_local_selection_ms
        + candidate_return_ms
        + coordinator_fusion_ms
    )
    full = _estimate_transfer_case(
        total_tokens,
        full_transfer_unit_count,
        full_transfer_unit_count,
        shared_query_compute_ms,
        cost_config,
    )

    active_selected = _estimate_transfer_case(
        transfer_accounting.get("selected_token_count", 0),
        transfer_accounting.get("unique_transfer_unit_count", 0),
        selected_segments,
        active_compute_time_ms,
        cost_config,
        parallel_fetch=True,
        node_token_counts=selected_node_token_counts,
        node_segment_counts=selected_node_segment_counts,
    )
    active_selected.update(
        {
            "query_q_prepare_ms": float(query_q_prepare_ms),
            "coarse_metadata_routing_ms": float(coarse_routing_ms),
            "query_dispatch_ms": float(query_dispatch_ms),
            "summary_metadata_fetch_ms": float(summary_metadata_fetch_ms),
            "node_local_selection_ms": float(node_local_selection_ms),
            "candidate_return_ms": float(candidate_return_ms),
            "coordinator_fusion_ms": float(coordinator_fusion_ms),
            "offline_summary_prepare_ms": float(
                routing_breakdown.get("node_summary_prepare_ms", 0.0)
            ),
            "oracle_exact_qk_ms_excluded": float(
                routing_breakdown.get("qk_total_stage_ms", 0.0)
            ),
            "summary_candidate_count": int(summary_candidate_count),
            "returned_candidate_count": int(returned_candidate_count),
            "summary_bytes": float(summary_bytes),
            "candidate_return_bytes": float(candidate_return_bytes),
        }
    )
    selected = active_selected if ttft_model == "active_node_v2" else legacy_selected

    def reduction(selected_value, full_value):
        if full_value <= 0:
            return 0.0
        return float(1.0 - (selected_value / full_value))

    return {
        "routing_overhead_ms": float(routing_overhead_ms),
        "routing_wall_clock_ms": float(
            routing_overhead_ms if routing_wall_clock_ms is None else routing_wall_clock_ms
        ),
        "routing_simulator_excluded_ms": float(simulator_excluded_ms),
        "assumptions": {
            "timing_model": ttft_model,
            "legacy_timing_model": "serial_compute_transfer_decode",
            "routing_compute_scope": (
                "active_node_online_excludes_offline_summary_prepare_and_oracle_exact_qk"
                if ttft_model == "active_node_v2"
                else "system_accounted_excludes_simulator_key_slicing"
            ),
            "num_hidden_layers": int(cost_config.get("num_hidden_layers", 0)),
            "num_key_value_heads": int(cost_config.get("num_key_value_heads", 0)),
            "head_dim": int(cost_config.get("head_dim", 0)),
            "kv_cache_dtype_bytes": float(cost_config.get("kv_cache_dtype_bytes", 0.0)),
            "kv_bytes_per_token": float(cost_config.get("kv_bytes_per_token", 0.0)),
            "fetch_bandwidth_gbps": float(cost_config.get("fetch_bandwidth_gbps", 0.0)),
            "control_bandwidth_gbps": float(cost_config.get("control_bandwidth_gbps", 0.0)),
            "per_node_rtt_ms": per_node_rtt_ms,
            "control_rtt_ms": control_rtt_ms,
            "per_segment_overhead_ms": per_segment_overhead_ms,
            "per_rpc_overhead_ms": per_rpc_overhead_ms,
            "decode_startup_ms": decode_startup_ms,
            "candidate_metadata_bytes": candidate_metadata_bytes,
            "query_request_bytes": query_request_bytes,
            "shared_query_compute_ms": float(shared_query_compute_ms),
            "full_baseline_compute_scope": (
                "includes_shared_query_q_prepare"
                if ttft_model == "active_node_v2"
                else "fetch_decode_only"
            ),
        },
        "selected": selected,
        "legacy_selected": legacy_selected,
        "active_node_selected": active_selected,
        "full": full,
        "reduction": {
            "kv_bytes_reduction_ratio": reduction(selected["kv_bytes"], full["kv_bytes"]),
            "fetch_latency_reduction_ratio": reduction(
                selected["fetch_latency_ms"],
                full["fetch_latency_ms"],
            ),
            "ttft_reduction_ratio": reduction(
                selected["estimated_ttft_ms"],
                full["estimated_ttft_ms"],
            ),
            "legacy_ttft_reduction_ratio": reduction(
                legacy_selected["estimated_ttft_ms"],
                full["estimated_ttft_ms"],
            ),
        },
    }

def _avg(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _avg_path(results, *path):
    vals = []
    for result in results:
        item = result
        for key in path:
            item = (item or {}).get(key)
        if item is not None:
            vals.append(float(item))
    return _avg(vals)


def summarize_transfer_metrics(results, key):
    items = [r.get(key) for r in results if r.get(key)]
    if not items:
        return None
    return {
        "avg_selected_chunks": _avg(i.get("selected_chunk_count", 0.0) for i in items),
        "avg_selected_tokens": _avg(i.get("selected_token_count", 0.0) for i in items),
        "avg_transfer_units": _avg(i.get("unique_transfer_unit_count", 0.0) for i in items),
        "avg_transfer_segments": _avg(i.get("transfer_segment_count", 0.0) for i in items),
        "avg_global_segments": _avg(
            i.get("global_contiguous_segment_count", 0.0) for i in items
        ),
        "avg_chunks_per_transfer_segment": _avg(
            i.get("avg_chunks_per_transfer_segment", 0.0) for i in items
        ),
        "avg_coalescing_gain": _avg(i.get("coalescing_gain_transfer", 0.0) for i in items),
    }


def summarize_system_cost_metrics(results, key):
    items = [r.get(key) for r in results if r.get(key)]
    if not items:
        return None
    first_assumptions = items[0].get("assumptions", {})
    return {
        "assumptions": first_assumptions,
        "avg_routing_overhead_ms": _avg(i.get("routing_overhead_ms", 0.0) for i in items),
        "avg_routing_wall_clock_ms": _avg(
            i.get("routing_wall_clock_ms", i.get("routing_overhead_ms", 0.0)) for i in items
        ),
        "avg_routing_simulator_excluded_ms": _avg(
            i.get("routing_simulator_excluded_ms", 0.0) for i in items
        ),
        "avg_selected_compute_time_ms": _avg_path(items, "selected", "compute_time_ms"),
        "avg_selected_kv_mib": _avg_path(items, "selected", "kv_mib"),
        "avg_full_kv_mib": _avg_path(items, "full", "kv_mib"),
        "avg_kv_bytes_reduction_ratio": _avg_path(items, "reduction", "kv_bytes_reduction_ratio"),
        "avg_selected_fetch_latency_ms": _avg_path(items, "selected", "fetch_latency_ms"),
        "avg_full_fetch_latency_ms": _avg_path(items, "full", "fetch_latency_ms"),
        "avg_fetch_latency_reduction_ratio": _avg_path(
            items,
            "reduction",
            "fetch_latency_reduction_ratio",
        ),
        "avg_selected_decode_time_ms": _avg_path(items, "selected", "decode_time_ms"),
        "avg_selected_ttft_ms": _avg_path(items, "selected", "estimated_ttft_ms"),
        "avg_full_ttft_ms": _avg_path(items, "full", "estimated_ttft_ms"),
        "avg_ttft_reduction_ratio": _avg_path(items, "reduction", "ttft_reduction_ratio"),
        "avg_legacy_selected_ttft_ms": _avg_path(
            items, "legacy_selected", "estimated_ttft_ms"
        ),
        "avg_legacy_ttft_reduction_ratio": _avg_path(
            items, "reduction", "legacy_ttft_reduction_ratio"
        ),
        "avg_query_dispatch_ms": _avg_path(items, "active_node_selected", "query_dispatch_ms"),
        "avg_summary_metadata_fetch_ms": _avg_path(
            items, "active_node_selected", "summary_metadata_fetch_ms"
        ),
        "avg_node_local_selection_ms": _avg_path(
            items, "active_node_selected", "node_local_selection_ms"
        ),
        "avg_candidate_return_ms": _avg_path(
            items, "active_node_selected", "candidate_return_ms"
        ),
        "avg_coordinator_fusion_ms": _avg_path(
            items, "active_node_selected", "coordinator_fusion_ms"
        ),
        "avg_offline_summary_prepare_ms": _avg_path(
            items, "active_node_selected", "offline_summary_prepare_ms"
        ),
        "avg_oracle_exact_qk_ms_excluded": _avg_path(
            items, "active_node_selected", "oracle_exact_qk_ms_excluded"
        ),
        "avg_parallel_fetch_latency_ms": _avg_path(
            items, "active_node_selected", "parallel_transfer_time_ms"
        ),
    }


def summarize_answer_metrics(results):
    items = [r.get("answer_eval") for r in results if r.get("answer_eval")]
    if not items:
        return None
    oracle_items = [
        item
        for item in items
        if bool(item.get("oracle_answer_available", True))
    ]
    return {
        "avg_full_answer_f1": _avg(i.get("full_answer_f1", 0.0) for i in items),
        "avg_selected_answer_f1": _avg(i.get("selected_answer_f1", 0.0) for i in items),
        "num_oracle_answer_cases": len(oracle_items),
        "oracle_answer_available_rate": len(oracle_items) / len(items)
        if items
        else 0.0,
        "avg_oracle_answer_f1": _avg(
            i.get("oracle_answer_f1", 0.0) for i in oracle_items
        ),
        "avg_answer_f1_delta": _avg(i.get("answer_f1_delta", 0.0) for i in items),
        "avg_oracle_answer_f1_delta_vs_full": _avg(
            i.get("oracle_answer_f1_delta_vs_full", 0.0) for i in oracle_items
        ),
        "avg_selected_answer_f1_delta_vs_oracle": _avg(
            i.get("selected_answer_f1_delta_vs_oracle", 0.0) for i in oracle_items
        ),
        "avg_full_context_tokens": _avg(i.get("full_context_tokens", 0.0) for i in items),
        "avg_selected_context_tokens": _avg(i.get("selected_context_tokens", 0.0) for i in items),
        "avg_oracle_context_tokens": _avg(
            i.get("oracle_context_tokens", 0.0) for i in oracle_items
        ),
        "avg_context_token_saving_ratio": _avg(
            i.get("context_token_saving_ratio", 0.0) for i in items
        ),
        "selected_ge_full_rate": _avg(
            1.0 if i.get("selected_answer_f1", 0.0) >= i.get("full_answer_f1", 0.0) else 0.0
            for i in items
        ),
        "oracle_ge_full_rate": _avg(
            1.0 if i.get("oracle_answer_f1", 0.0) >= i.get("full_answer_f1", 0.0) else 0.0
            for i in oracle_items
        ),
        "selected_ge_oracle_rate": _avg(
            1.0 if i.get("selected_answer_f1", 0.0) >= i.get("oracle_answer_f1", 0.0) else 0.0
            for i in oracle_items
        ),
        "full_bad_output_rate": _avg(1.0 if i.get("full_bad_output") else 0.0 for i in items),
        "selected_bad_output_rate": _avg(
            1.0 if i.get("selected_bad_output") else 0.0 for i in items
        ),
        "oracle_bad_output_rate": _avg(
            1.0 if i.get("oracle_bad_output") else 0.0 for i in oracle_items
        ),
    }


def summarize_cachegen_full_metrics(results):
    items = [r.get("cachegen_full_estimate") for r in results if r.get("cachegen_full_estimate")]
    if not items:
        return None
    ok_items = [item for item in items if item.get("status") == "ok"]
    if not ok_items:
        first = items[0]
        return {
            "status": "error",
            "n_cachegen_cases": len(items),
            "n_cachegen_ok": 0,
            "first_error_type": first.get("error_type", ""),
            "first_error": first.get("error", ""),
        }
    return {
        "status": "ok",
        "n_cachegen_cases": len(items),
        "n_cachegen_ok": len(ok_items),
        "baseline_type": "cachegen_full_estimated",
        "avg_compressed_mib": _avg(i.get("compressed_mib", 0.0) for i in ok_items),
        "avg_compression_saving_ratio": _avg(
            i.get("compression_saving_ratio", 0.0) for i in ok_items
        ),
        "avg_cachegen_total_encode_ms": _avg(i.get("total_encode_ms", 0.0) for i in ok_items),
        "avg_cachegen_fetch_latency_ms": _avg(i.get("fetch_latency_ms", 0.0) for i in ok_items),
        "avg_cachegen_ttft_ms": _avg(i.get("estimated_ttft_ms", 0.0) for i in ok_items),
        "avg_selected_vs_cachegen_ttft_delta_ms": _avg(
            i.get("selected_vs_cachegen_ttft_delta_ms", 0.0) for i in ok_items
        ),
        "avg_selected_vs_cachegen_ttft_saving_ratio": _avg(
            i.get("selected_vs_cachegen_ttft_saving_ratio", 0.0) for i in ok_items
        ),
        "avg_cachegen_estimated_answer_f1": _avg(
            i.get("estimated_answer_f1", 0.0) for i in ok_items
        ),
        "avg_selected_answer_f1_delta_vs_cachegen": _avg(
            i.get("selected_answer_f1_delta_vs_cachegen", 0.0) for i in ok_items
        ),
    }


def summarize_cachegen_roundtrip_answer_metrics(results):
    items = [
        r.get("cachegen_roundtrip_answer")
        for r in results
        if r.get("cachegen_roundtrip_answer")
    ]
    if not items:
        return None
    ok_items = [item for item in items if item.get("status") == "ok"]
    if not ok_items:
        first = items[0]
        return {
            "status": "error",
            "n_cachegen_roundtrip_cases": len(items),
            "n_cachegen_roundtrip_ok": 0,
            "first_error_type": first.get("error_type", first.get("status", "")),
            "first_error": first.get("error", first.get("skip_reason", "")),
        }
    return {
        "status": "ok",
        "n_cachegen_roundtrip_cases": len(items),
        "n_cachegen_roundtrip_ok": len(ok_items),
        "baseline_type": "cachegen_full_roundtrip_answer",
        "avg_answer_f1": _avg(i.get("answer_f1", 0.0) for i in ok_items),
        "avg_answer_f1_delta_vs_full": _avg(
            i.get("answer_f1_delta_vs_full", 0.0) for i in ok_items
        ),
        "avg_answer_f1_delta_vs_selected": _avg(
            i.get("answer_f1_delta_vs_selected", 0.0) for i in ok_items
        ),
        "avg_selected_answer_f1_delta_vs_cachegen_roundtrip": _avg(
            i.get("selected_answer_f1_delta_vs_cachegen_roundtrip", 0.0)
            for i in ok_items
        ),
        "bad_output_rate": _avg(1.0 if i.get("bad_output") else 0.0 for i in ok_items),
        "initial_bad_output_rate": _avg(
            1.0 if i.get("initial_bad_output") else 0.0 for i in ok_items
        ),
        "retry_rate": _avg(1.0 if i.get("retried") else 0.0 for i in ok_items),
        "retry_used_rate": _avg(1.0 if i.get("used_retry") else 0.0 for i in ok_items),
        "avg_prompt_tokens": _avg(i.get("prompt_tokens", 0.0) for i in ok_items),
        "avg_compressed_mib": _avg(i.get("compressed_mib", 0.0) for i in ok_items),
        "avg_compression_saving_ratio": _avg(
            i.get("compression_saving_ratio", 0.0) for i in ok_items
        ),
        "avg_prefill_ms": _avg(i.get("prefill_ms", 0.0) for i in ok_items),
        "avg_prepare_ms": _avg(i.get("prepare_ms", 0.0) for i in ok_items),
        "avg_encode_ms": _avg(i.get("encode_ms", 0.0) for i in ok_items),
        "avg_total_encode_ms": _avg(i.get("total_encode_ms", 0.0) for i in ok_items),
        "avg_decode_ms": _avg(i.get("measured_decode_ms", 0.0) for i in ok_items),
        "avg_generation_ms": _avg(i.get("generation_ms", 0.0) for i in ok_items),
        "avg_fetch_latency_ms": _avg(i.get("fetch_latency_ms", 0.0) for i in ok_items),
        "avg_ttft_ms": _avg(i.get("estimated_ttft_ms", 0.0) for i in ok_items),
        "avg_selected_vs_cachegen_roundtrip_ttft_delta_ms": _avg(
            i.get("selected_vs_cachegen_roundtrip_ttft_delta_ms", 0.0)
            for i in ok_items
        ),
        "avg_selected_vs_cachegen_roundtrip_ttft_saving_ratio": _avg(
            i.get("selected_vs_cachegen_roundtrip_ttft_saving_ratio", 0.0)
            for i in ok_items
        ),
    }


def summarize_routing_timing_metrics(results):
    items = [r.get("routing_timing") for r in results if r.get("routing_timing")]
    if not items:
        return None
    steady_items = [item for item in items if not bool(item.get("timing_is_first_query", False))]

    def avg(key, source=None):
        source = items if source is None else source
        return _avg(float(item.get(key, 0.0)) for item in source)

    qk_infos = [r.get("qk_route_info") or {} for r in results]
    summary = {
        "num_timing_cases": len(items),
        "num_steady_state_timing_cases": len(steady_items),
        "avg_offline_route_artifact_prep_ms": avg("offline_route_artifact_prep_ms"),
        "avg_online_route_decision_ms": avg("online_route_decision_ms"),
        "avg_system_online_route_decision_ms": avg("system_online_route_decision_ms"),
        "avg_simulator_key_prepare_excluded_ms": avg("simulator_key_prepare_excluded_ms"),
        "avg_query_tokenize_ms": avg("query_tokenize_ms"),
        "avg_coarse_topic_routing_ms": avg("coarse_topic_routing_ms"),
        "avg_candidate_key_prepare_ms": avg("candidate_key_prepare_ms"),
        "avg_query_q_prepare_ms": avg("query_q_prepare_ms"),
        "avg_candidate_prefilter_ms": avg("candidate_prefilter_ms"),
        "avg_dynamic_candidate_pool_ms": avg("dynamic_candidate_pool_ms"),
        "avg_coarse_segment_gate_ms": avg("coarse_segment_gate_ms"),
        "avg_node_summary_gate_ms": avg("node_summary_gate_ms"),
        "avg_node_summary_prepare_ms": avg("node_summary_prepare_ms"),
        "avg_node_summary_score_ms": avg("node_summary_score_ms"),
        "avg_node_summary_aggregate_ms": avg("node_summary_aggregate_ms"),
        "avg_qk_model_inference_ms": avg("qk_model_inference_ms"),
        "avg_qk_score_aggregation_ms": avg("qk_score_aggregation_ms"),
        "avg_qk_scoring_ms": avg("qk_scoring_ms"),
        "avg_qk_total_stage_ms": avg("qk_total_stage_ms"),
        "avg_selection_postprocess_ms": avg("selection_postprocess_ms"),
        "avg_route_unaccounted_ms": avg("route_unaccounted_ms"),
        "avg_candidates_scored_qk": _avg(
            float(info.get("num_candidates_scored_qk", 0.0) or 0.0) for info in qk_infos
        ),
        "avg_adaptive_topic_rescue_trigger_rate": _avg(
            1.0 if info.get("adaptive_topic_rescue_triggered", False) else 0.0
            for info in qk_infos
        ),
        "avg_adaptive_topic_rescue_final_k": _avg(
            float(info.get("adaptive_topic_rescue_final_k", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_adaptive_topic_rescue_gap_ratio": _avg(
            float(info.get("adaptive_topic_rescue_gap_ratio", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_adaptive_topic_rescue_top1_ratio": _avg(
            float(info.get("adaptive_topic_rescue_top1_ratio", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_dynamic_candidate_pool_prune_ratio": _avg(
            float(info.get("dynamic_candidate_pool_prune_ratio", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_candidate_prefilter_adaptive_keep_ratio": _avg(
            float(info.get("candidate_prefilter_adaptive_keep_ratio", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_candidate_prefilter_adaptive_uncertainty": _avg(
            float(info.get("candidate_prefilter_adaptive_uncertainty", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_coarse_segment_gate_adaptive_keep_ratio": _avg(
            float(info.get("coarse_segment_gate_adaptive_keep_ratio", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_coarse_segment_gate_adaptive_uncertainty": _avg(
            float(info.get("coarse_segment_gate_adaptive_uncertainty", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_node_summary_gate_before": _avg(
            float(info.get("node_summary_gate_before", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_node_summary_gate_after": _avg(
            float(info.get("node_summary_gate_after", 0.0) or 0.0)
            for info in qk_infos
        ),
        "avg_node_summary_gate_prune_ratio": _avg(
            float(info.get("node_summary_gate_prune_ratio", 0.0) or 0.0)
            for info in qk_infos
        ),
    }
    for key in [
        "online_route_decision_ms",
        "system_online_route_decision_ms",
        "simulator_key_prepare_excluded_ms",
        "query_tokenize_ms",
        "qk_model_inference_ms",
        "qk_scoring_ms",
        "qk_total_stage_ms",
    ]:
        summary[f"avg_{key}_steady_state"] = avg(key, steady_items)
    return summary


def summarize_relevant_candidate_survival(results):
    items = [
        (r.get("qk_route_info") or {}).get("relevant_candidate_survival")
        for r in results
        if (r.get("qk_route_info") or {}).get("relevant_candidate_survival")
    ]
    if not items:
        return None

    failure_counts = defaultdict(int)
    first_drop_counts = defaultdict(int)
    first_zero_counts = defaultdict(int)
    oracle_diagnosis_counts = defaultdict(int)
    for item in items:
        failure_counts[str(item.get("failure_stage", "unknown"))] += 1
        first_drop_counts[str(item.get("first_drop_stage", "unknown"))] += 1
        first_zero_counts[str(item.get("first_zero_stage", "unknown"))] += 1
        oracle_diagnosis_counts[str(item.get("qk_oracle_diagnosis", "unknown"))] += 1

    return {
        "n_survival_cases": len(items),
        "avg_candidate_build_turn_recall": _avg(
            item.get("candidate_build_turn_recall", 0.0) for item in items
        ),
        "avg_topic_filter_turn_recall": _avg(
            item.get("topic_filter_turn_recall", 0.0) for item in items
        ),
        "avg_candidate_prefilter_turn_recall": _avg(
            item.get("candidate_prefilter_turn_recall", 0.0) for item in items
        ),
        "avg_dynamic_candidate_pool_turn_recall": _avg(
            item.get("dynamic_candidate_pool_turn_recall", 0.0) for item in items
        ),
        "avg_coarse_segment_gate_turn_recall": _avg(
            item.get("coarse_segment_gate_turn_recall", 0.0) for item in items
        ),
        "avg_node_summary_gate_turn_recall": _avg(
            item.get("node_summary_gate_turn_recall", 0.0) for item in items
        ),
        "avg_qk_scored_turn_recall": _avg(
            item.get("qk_scored_turn_recall", 0.0) for item in items
        ),
        "avg_qk_selected_turn_recall": _avg(
            item.get("qk_selected_turn_recall", 0.0) for item in items
        ),
        "avg_qk_selection_budget": _avg(
            item.get("qk_selection_budget", 0.0) for item in items
        ),
        "avg_qk_ranked_candidate_count": _avg(
            item.get("qk_ranked_candidate_count", 0.0) for item in items
        ),
        "avg_qk_scored_relevant_turn_count": _avg(
            item.get("qk_scored_relevant_turn_count", 0.0) for item in items
        ),
        "avg_qk_ranked_top_budget_turn_recall": _avg(
            item.get("qk_ranked_top_budget_turn_recall", 0.0) for item in items
        ),
        "qk_oracle_turn_hit_at_selected_budget_rate": _avg(
            1.0 if item.get("qk_oracle_turn_hit_at_selected_budget", False) else 0.0
            for item in items
        ),
        "avg_qk_oracle_turn_recall_at_selected_budget": _avg(
            item.get("qk_oracle_turn_recall_at_selected_budget", 0.0)
            for item in items
        ),
        "avg_qk_selector_gap": _avg(
            item.get("qk_selector_gap", 0.0) for item in items
        ),
        "avg_qk_budget_cap_gap": _avg(
            item.get("qk_budget_cap_gap", 0.0) for item in items
        ),
        "avg_qk_first_relevant_rank": _avg(
            item.get("qk_first_relevant_rank", 0.0) for item in items
        ),
        "avg_qk_first_relevant_rank_pct": _avg(
            item.get("qk_first_relevant_rank_pct", 0.0) for item in items
        ),
        "avg_qk_relevant_turn_best_rank_mean": _avg(
            item.get("qk_relevant_turn_best_rank_mean", 0.0) for item in items
        ),
        "avg_qk_relevant_turn_best_rank_median": _avg(
            item.get("qk_relevant_turn_best_rank_median", 0.0) for item in items
        ),
        "avg_turn_utility_first_relevant_rank": _avg(
            item.get("turn_utility_first_relevant_rank", 0.0) for item in items
        ),
        "avg_turn_utility_first_relevant_rank_pct": _avg(
            item.get("turn_utility_first_relevant_rank_pct", 0.0) for item in items
        ),
        "avg_turn_utility_mean_gold_rank": _avg(
            item.get("turn_utility_mean_gold_rank", 0.0) for item in items
        ),
        "avg_turn_utility_recall_at_selected_budget": _avg(
            item.get("turn_utility_recall_at_selected_budget", 0.0)
            for item in items
        ),
        "avg_turn_utility_gap_to_qk_oracle": _avg(
            item.get("turn_utility_gap_to_qk_oracle", 0.0) for item in items
        ),
        "avg_turn_utility_gap_to_scored": _avg(
            item.get("turn_utility_gap_to_scored", 0.0) for item in items
        ),
        "avg_turn_utility_ranked_turn_count": _avg(
            item.get("turn_utility_ranked_turn_count", 0.0) for item in items
        ),
        "failure_stage_counts": dict(sorted(failure_counts.items())),
        "first_drop_stage_counts": dict(sorted(first_drop_counts.items())),
        "first_zero_stage_counts": dict(sorted(first_zero_counts.items())),
        "qk_oracle_diagnosis_counts": dict(sorted(oracle_diagnosis_counts.items())),
    }


def summarize_route_selection_debug(results):
    items = [
        (r.get("qk_route_info") or {}).get("route_selection_debug")
        for r in results
        if (r.get("qk_route_info") or {}).get("route_selection_debug")
    ]
    if not items:
        return None

    def avg_key(key):
        return _avg(float(item.get(key, 0.0) or 0.0) for item in items)

    return {
        "n_selection_debug_cases": len(items),
        "avg_pack_anchor_count": avg_key("anchor_count"),
        "avg_pack_support_budget": avg_key("support_budget"),
        "avg_pack_coverage_budget": avg_key("coverage_budget"),
        "avg_pack_neighbor_support_added": avg_key("neighbor_support_added"),
        "avg_pack_same_turn_support_added": avg_key("same_turn_support_added"),
        "avg_pack_adjacent_turn_support_added": avg_key(
            "adjacent_turn_support_added"
        ),
        "avg_pack_support_total_added": avg_key("support_total_added"),
        "avg_pack_coverage_turn_added": avg_key("coverage_turn_added"),
        "avg_pack_qk_backfill_added": avg_key("qk_backfill_added"),
        "avg_pack_support_candidates_seen": avg_key("support_candidates_seen"),
        "avg_pack_support_candidates_below_threshold": avg_key(
            "support_candidates_below_threshold"
        ),
        "avg_pack_replacement_budget": avg_key("replacement_budget"),
        "avg_pack_replacement_added": avg_key("replacement_added"),
        "avg_pack_replacement_skipped_no_drop": avg_key(
            "replacement_skipped_no_drop"
        ),
        "avg_pack_replacement_skipped_low_score": avg_key(
            "replacement_skipped_low_score"
        ),
        "avg_pack_replacement_score_tolerance": avg_key(
            "replacement_score_tolerance"
        ),
        "avg_pack_replacement_score_span": avg_key("replacement_score_span"),
        "avg_pack_replacement_median_gap": avg_key("replacement_median_gap"),
        "avg_unique_turn_added": avg_key("unique_turn_added"),
        "avg_global_unique_turn_added": avg_key("global_unique_turn_added"),
        "avg_duplicate_deferred": avg_key("duplicate_deferred"),
        "avg_duplicate_backfill_added": avg_key("duplicate_backfill_added"),
        "avg_selection_selected_turn_count": avg_key("selected_turn_count"),
        "avg_selection_duplicate_chunk_count": avg_key(
            "selected_duplicate_chunk_count"
        ),
        "avg_turn_utility_ranked_turn_count": avg_key("candidate_turns_before"),
    }


def display_strategy_name(strategy_name, args):
    if str(strategy_name) == "qk" and getattr(args, "routing_granularity", "") == "hierarchical":
        return "qk_restricted"
    return str(strategy_name)


def build_summary_payload(results, args, num_nodes):
    n = len(results)
    avg_variance = _avg(r["qk_score_variance"] for r in results)
    avg_range = _avg(r["qk_score_range"] for r in results)
    high_var = sum(1 for r in results if r["qk_score_variance"] > 0.001)
    avg_turns = _avg(r["num_turns"] for r in results)
    max_route_units = max((int(r.get("num_route_units", 0)) for r in results), default=0)
    unit_label = results[0].get("routing_unit_type", "nodes")
    unit_singular = unit_label[:-1] if unit_label.endswith("s") else unit_label
    avg_relevant_nodes = _avg(len(r["relevant_nodes"]) for r in results)
    all_node_covered = sum(
        1 for r in results if len(r["relevant_nodes"]) == r.get("num_route_units", num_nodes)
    )
    strategy_keys = list(results[0]["strategy_results"].keys())

    def top_node_dist(strategy_name):
        dist = defaultdict(int)
        for result in results:
            dist[result["strategy_results"][strategy_name]["top_node"]] += 1
        return dict(sorted(dist.items()))

    per_strategy = {}
    for sname in strategy_keys:
        top1_any = sum(
            1
            for r in results
            if r["routing_eval"]["strategy_hits"].get(sname, {}).get("top1_any_relevant_hit")
        )
        top2_any = sum(
            1
            for r in results
            if r["routing_eval"]["strategy_hits"].get(sname, {}).get("top2_any_relevant_hit")
        )
        top1_dom = sum(
            1
            for r in results
            if r["routing_eval"]["strategy_hits"].get(sname, {}).get("top1_dominant_hit")
        )
        top2_dom = sum(
            1
            for r in results
            if r["routing_eval"]["strategy_hits"].get(sname, {}).get("top2_dominant_hit")
        )
        per_strategy[sname] = {
            "top1_any_hits": top1_any,
            "top2_any_hits": top2_any,
            "top1_any_hit_rate": top1_any / n if n else 0.0,
            "top2_any_hit_rate": top2_any / n if n else 0.0,
            "top1_dominant_hits": top1_dom,
            "top2_dominant_hits": top2_dom,
            "top1_dominant_hit_rate": top1_dom / n if n else 0.0,
            "top2_dominant_hit_rate": top2_dom / n if n else 0.0,
        }

    selected_node_hits = sum(
        1 for r in results if r["routing_eval"].get("route_selected_unit_hit", False)
    )
    selected_turn_hits = sum(
        1 for r in results if r["routing_eval"].get("route_selected_turn_hit", False)
    )
    transfer_summary = summarize_transfer_metrics(results, "transfer_accounting") or {}
    virtual_node_transfer_summary = summarize_transfer_metrics(
        results,
        "virtual_node_transfer_accounting",
    )
    steady_results = [
        r
        for r in results
        if not bool((r.get("routing_timing") or {}).get("timing_is_first_query", False))
    ]
    return {
        "n": n,
        "avg_variance": float(avg_variance),
        "avg_range": float(avg_range),
        "high_var": int(high_var),
        "avg_turns": float(avg_turns),
        "max_route_units": int(max_route_units),
        "unit_label": unit_label,
        "unit_singular": unit_singular,
        "avg_relevant_nodes": float(avg_relevant_nodes),
        "all_node_covered": int(all_node_covered),
        "strategy_keys": strategy_keys,
        "top_node_distributions": {sname: top_node_dist(sname) for sname in strategy_keys},
        "per_strategy": per_strategy,
        "selected_node_hits": int(selected_node_hits),
        "selected_turn_hits": int(selected_turn_hits),
        "avg_selected_turn_recall": _avg(
            r["routing_eval"].get("route_selected_turn_recall", 0.0) for r in results
        ),
        "avg_selected_turn_precision": _avg(
            r["routing_eval"].get("route_selected_turn_precision", 0.0) for r in results
        ),
        "avg_selected_turn_f1": _avg(
            r["routing_eval"].get("route_selected_turn_f1", 0.0) for r in results
        ),
        "avg_selected_chunks": float(transfer_summary.get("avg_selected_chunks", 0.0)),
        "avg_selected_tokens": float(transfer_summary.get("avg_selected_tokens", 0.0)),
        "avg_transfer_units": float(transfer_summary.get("avg_transfer_units", 0.0)),
        "avg_transfer_segments": float(transfer_summary.get("avg_transfer_segments", 0.0)),
        "avg_global_segments": float(transfer_summary.get("avg_global_segments", 0.0)),
        "avg_chunks_per_transfer_segment": float(
            transfer_summary.get("avg_chunks_per_transfer_segment", 0.0)
        ),
        "avg_coalescing_gain": float(transfer_summary.get("avg_coalescing_gain", 0.0)),
        "virtual_node_transfer_summary": virtual_node_transfer_summary,
        "avg_virtual_node_transfer_units": float(
            (virtual_node_transfer_summary or {}).get("avg_transfer_units", 0.0)
        ),
        "avg_virtual_node_transfer_segments": float(
            (virtual_node_transfer_summary or {}).get("avg_transfer_segments", 0.0)
        ),
        "avg_virtual_node_chunks_per_transfer_segment": float(
            (virtual_node_transfer_summary or {}).get("avg_chunks_per_transfer_segment", 0.0)
        ),
        "avg_virtual_node_coalescing_gain": float(
            (virtual_node_transfer_summary or {}).get("avg_coalescing_gain", 0.0)
        ),
        "system_cost_summary": summarize_system_cost_metrics(results, "system_cost"),
        "system_cost_steady_state_summary": (
            summarize_system_cost_metrics(steady_results, "system_cost")
            if steady_results
            else None
        ),
        "cachegen_full_summary": summarize_cachegen_full_metrics(results),
        "cachegen_roundtrip_answer_summary": (
            summarize_cachegen_roundtrip_answer_metrics(results)
        ),
        "routing_timing_summary": summarize_routing_timing_metrics(results),
        "relevant_candidate_survival_summary": summarize_relevant_candidate_survival(results),
        "route_selection_debug_summary": summarize_route_selection_debug(results),
        "answer_summary": summarize_answer_metrics(results),
    }


def print_summary(summary_payload, args, num_nodes):
    n = summary_payload["n"]
    unit_label = summary_payload["unit_label"]
    unit_singular = summary_payload["unit_singular"]

    print(f"\n{'=' * 70}")
    print(f"QMSum routing summary (n={n})")
    print(f"{'=' * 70}")
    print(f"  avg turns:              {summary_payload['avg_turns']:.1f}")
    print(f"  routing units:          {unit_label}")
    if args.routing_granularity in ["node", "chunk"]:
        print(f"  nodes:                  {num_nodes}")
        print(f"  node assignment:        {args.node_assignment_mode}")
    else:
        print(f"  topic nodes observed:   up to {summary_payload['max_route_units']}")
        print("  top-level assignment:   qmsum topic_list spans")
    print(f"  Q-K avg score variance: {summary_payload['avg_variance']:.6f}")
    print(f"  Q-K avg score range:    {summary_payload['avg_range']:.4f}")
    print(
        f"  distinguishable:        {summary_payload['high_var']}/{n} "
        f"({100 * summary_payload['high_var'] / n:.0f}%)"
    )
    print(f"  avg relevant {unit_label}: {summary_payload['avg_relevant_nodes']:.2f}")
    print(
        f"  all-{unit_singular} relevant span: {summary_payload['all_node_covered']}/{n} "
        f"({100 * summary_payload['all_node_covered'] / n:.1f}%)"
    )

    unit_prefix = "T" if unit_label == "topics" else "N"
    print(f"\n  Top-{unit_singular} distribution:")
    for sname in sorted(summary_payload["strategy_keys"]):
        dist = summary_payload["top_node_distributions"][sname]
        dist_str = ", ".join(f"{unit_prefix}{k}:{v}" for k, v in dist.items())
        total_units_seen = max(dist.keys(), default=-1) + 1 if dist else num_nodes
        shown_name = display_strategy_name(sname, args)
        print(
            f"    {shown_name:>14}: {dist_str} "
            f"(hit {len(dist)}/{total_units_seen} {unit_label})"
        )

    print(f"\n  Relevant-span {unit_singular} hit rate (loose):")
    for sname in sorted(summary_payload["per_strategy"]):
        sm = summary_payload["per_strategy"][sname]
        shown_name = display_strategy_name(sname, args)
        print(
            f"    {shown_name:>14} top-1: {sm['top1_any_hits']}/{n} "
            f"({100 * sm['top1_any_hit_rate']:.1f}%), "
            f"top-2: {sm['top2_any_hits']}/{n} "
            f"({100 * sm['top2_any_hit_rate']:.1f}%)"
        )
    print(f"\n  Dominant-{unit_singular} hit rate (stricter):")
    for sname in sorted(summary_payload["per_strategy"]):
        sm = summary_payload["per_strategy"][sname]
        shown_name = display_strategy_name(sname, args)
        print(
            f"    {shown_name:>14} top-1: {sm['top1_dominant_hits']}/{n} "
            f"({100 * sm['top1_dominant_hit_rate']:.1f}%), "
            f"top-2: {sm['top2_dominant_hits']}/{n} "
            f"({100 * sm['top2_dominant_hit_rate']:.1f}%)"
        )

    print("\n  Q-K selected evidence quality:")
    selected_strategy = args.hier_top_strategy if args.routing_granularity == "hierarchical" else "qk"
    print(
        f"    selected-{unit_singular} hit ({selected_strategy}): "
        f"{summary_payload['selected_node_hits']}/{n} "
        f"({100 * summary_payload['selected_node_hits'] / n:.1f}%)"
    )
    print(
        f"    selected-turn hit:   {summary_payload['selected_turn_hits']}/{n} "
        f"({100 * summary_payload['selected_turn_hits'] / n:.1f}%)"
    )
    print(f"    avg turn recall:     {100 * summary_payload['avg_selected_turn_recall']:.1f}%")
    print(f"    avg turn precision:  {100 * summary_payload['avg_selected_turn_precision']:.1f}%")
    print(f"    avg turn F1:         {100 * summary_payload['avg_selected_turn_f1']:.1f}%")

    survival = summary_payload.get("relevant_candidate_survival_summary")
    if survival:
        print("\n  Relevant candidate survival:")
        print(
            f"    topic-filter recall:   "
            f"{100 * survival['avg_topic_filter_turn_recall']:.1f}%"
        )
        print(
            f"    prefilter recall:      "
            f"{100 * survival['avg_candidate_prefilter_turn_recall']:.1f}%"
        )
        print(
            f"    dynamic-pool recall:   "
            f"{100 * survival['avg_dynamic_candidate_pool_turn_recall']:.1f}%"
        )
        print(
            f"    coarse-gate recall:    "
            f"{100 * survival['avg_coarse_segment_gate_turn_recall']:.1f}%"
        )
        print(
            f"    node-summary recall:   "
            f"{100 * survival.get('avg_node_summary_gate_turn_recall', 0.0):.1f}%"
        )
        print(
            f"    Q-K scored recall:     "
            f"{100 * survival['avg_qk_scored_turn_recall']:.1f}%"
        )
        print(
            f"    Q-K selected recall:   "
            f"{100 * survival['avg_qk_selected_turn_recall']:.1f}%"
        )
        print(
            f"    oracle@selected-budget:"
            f"{100 * survival.get('avg_qk_oracle_turn_recall_at_selected_budget', 0.0):.1f}% "
            f"(gap {100 * survival.get('avg_qk_selector_gap', 0.0):.1f}%)"
        )
        print(
            f"    first gold Q-K rank:   "
            f"{survival.get('avg_qk_first_relevant_rank', 0.0):.1f} / "
            f"{survival.get('avg_qk_ranked_candidate_count', 0.0):.1f}"
        )
        if str(getattr(args, "route_selection_mode", "")) in [
            "turn_utility",
            "turn_rank_fusion",
        ]:
            selector_label = (
                "turn-rank-fusion"
                if str(getattr(args, "route_selection_mode", "")) == "turn_rank_fusion"
                else "turn-utility"
            )
            print(
                f"    {selector_label}@budget: "
                f"{100 * survival.get('avg_turn_utility_recall_at_selected_budget', 0.0):.1f}% "
                f"(vs Q-K oracle gap "
                f"{100 * survival.get('avg_turn_utility_gap_to_qk_oracle', 0.0):.1f}%)"
            )
            print(
                f"    first gold {selector_label} rank: "
                f"{survival.get('avg_turn_utility_first_relevant_rank', 0.0):.1f} / "
                f"{survival.get('avg_turn_utility_ranked_turn_count', 0.0):.1f}"
            )
        failure_counts = survival.get("failure_stage_counts", {})
        if failure_counts:
            failure_str = ", ".join(
                f"{stage}:{count}" for stage, count in failure_counts.items()
            )
            print(f"    zero-recall stage mix:  {failure_str}")
        oracle_counts = survival.get("qk_oracle_diagnosis_counts", {})
        if oracle_counts:
            oracle_str = ", ".join(
                f"{stage}:{count}" for stage, count in oracle_counts.items()
            )
            print(f"    oracle diagnosis mix:  {oracle_str}")

    print("\n  KVDirect-style transfer accounting:")
    print(f"    avg selected chunks: {summary_payload['avg_selected_chunks']:.1f}")
    print(f"    avg selected tokens: {summary_payload['avg_selected_tokens']:.1f}")
    print(f"    avg transfer {unit_label}: {summary_payload['avg_transfer_units']:.2f}")
    print(f"    avg transfer segments: {summary_payload['avg_transfer_segments']:.2f}")
    print(f"    avg global segments:   {summary_payload['avg_global_segments']:.2f}")
    print(
        f"    avg chunks/segment:   "
        f"{summary_payload['avg_chunks_per_transfer_segment']:.2f}"
    )
    print(f"    avg coalescing gain:  {100 * summary_payload['avg_coalescing_gain']:.1f}%")

    virtual = summary_payload.get("virtual_node_transfer_summary")
    if virtual:
        print("\n  Virtual-node transfer accounting:")
        print(f"    avg transfer virtual_nodes: {virtual['avg_transfer_units']:.2f}")
        print(f"    avg node segments:    {virtual['avg_transfer_segments']:.2f}")
        print(f"    avg node chunks/seg:  {virtual['avg_chunks_per_transfer_segment']:.2f}")
        print(f"    avg node coalescing:  {100 * virtual['avg_coalescing_gain']:.1f}%")
    if getattr(args, "routing_granularity", "") == "hierarchical":
        print(
            "    note: qk_restricted means Q-K topic scores only within the lexical-selected topics"
        )

    selection_debug = summary_payload.get("route_selection_debug_summary")
    if selection_debug and str(getattr(args, "route_selection_mode", "")) in [
        "evidence_pack",
        "evidence_pack_v2",
        "evidence_pack_v3",
    ]:
        print("\n  Evidence-pack selection debug:")
        print(
            f"    avg anchors:         "
            f"{selection_debug['avg_pack_anchor_count']:.2f}"
        )
        if str(getattr(args, "route_selection_mode", "")) in [
            "evidence_pack_v2",
            "evidence_pack_v3",
        ]:
            print(
                f"    avg support budget:  "
                f"{selection_debug['avg_pack_support_budget']:.2f}"
            )
            print(
                f"    avg coverage budget: "
                f"{selection_debug['avg_pack_coverage_budget']:.2f}"
            )
        print(
            f"    avg local support:   "
            f"neighbor {selection_debug['avg_pack_neighbor_support_added']:.2f}, "
            f"same-turn {selection_debug['avg_pack_same_turn_support_added']:.2f}, "
            f"adjacent-turn {selection_debug['avg_pack_adjacent_turn_support_added']:.2f}"
        )
        print(
            f"    avg support total:   "
            f"{selection_debug['avg_pack_support_total_added']:.2f}"
        )
        print(
            f"    avg coverage/backfill:"
            f"{selection_debug['avg_pack_coverage_turn_added']:.2f} / "
            f"{selection_debug['avg_pack_qk_backfill_added']:.2f}"
        )
        if str(getattr(args, "route_selection_mode", "")) == "evidence_pack_v3":
            print(
                f"    avg replacements:    "
                f"{selection_debug['avg_pack_replacement_added']:.2f} / "
                f"{selection_debug['avg_pack_replacement_budget']:.2f}"
            )
    if selection_debug and str(getattr(args, "route_selection_mode", "")) in [
        "turn_utility",
        "turn_rank_fusion",
        "turn_unique",
        "turn_unique_guard",
        "turn_unique_soft",
    ]:
        if str(getattr(args, "route_selection_mode", "")) in [
            "turn_utility",
            "turn_rank_fusion",
        ]:
            selector_title = (
                "Turn-rank-fusion"
                if str(getattr(args, "route_selection_mode", "")) == "turn_rank_fusion"
                else "Turn-utility"
            )
            print(f"\n  {selector_title} selection debug:")
        else:
            print("\n  Turn-unique selection debug:")
            print(
                f"    avg unique turns added: "
                f"{selection_debug['avg_unique_turn_added']:.2f} core, "
                f"{selection_debug['avg_global_unique_turn_added']:.2f} global"
            )
            print(
                f"    avg duplicate deferred/backfill: "
                f"{selection_debug['avg_duplicate_deferred']:.2f} / "
                f"{selection_debug['avg_duplicate_backfill_added']:.2f}"
            )
        print(
            f"    avg selected turns/duplicate chunks: "
            f"{selection_debug['avg_selection_selected_turn_count']:.2f} / "
            f"{selection_debug['avg_selection_duplicate_chunk_count']:.2f}"
        )
        if str(getattr(args, "route_selection_mode", "")) in [
            "turn_utility",
            "turn_rank_fusion",
        ]:
            print(
                f"    avg ranked candidate turns: "
                f"{selection_debug['avg_turn_utility_ranked_turn_count']:.2f}"
            )
        if str(getattr(args, "route_selection_mode", "")) in [
            "turn_unique_guard",
            "turn_unique_soft",
        ]:
            print(
                f"    avg replacements: "
                f"{selection_debug['avg_pack_replacement_added']:.2f} / "
                f"{selection_debug['avg_pack_replacement_budget']:.2f}"
            )
            print(
                f"    avg skipped no-drop/low-score: "
                f"{selection_debug['avg_pack_replacement_skipped_no_drop']:.2f} / "
                f"{selection_debug['avg_pack_replacement_skipped_low_score']:.2f}"
            )
        if str(getattr(args, "route_selection_mode", "")) == "turn_unique_soft":
            print(
                f"    avg soft band span/tolerance/gap: "
                f"{selection_debug['avg_pack_replacement_score_span']:.4f} / "
                f"{selection_debug['avg_pack_replacement_score_tolerance']:.4f} / "
                f"{selection_debug['avg_pack_replacement_median_gap']:.4f}"
            )

    timing = summary_payload.get("routing_timing_summary")
    if timing:
        print("\n  Routing timing breakdown:")
        print(f"    avg offline prep:     {timing['avg_offline_route_artifact_prep_ms']:.2f} ms")
        print(
            f"    avg online routing wall:"
            f"{timing['avg_online_route_decision_ms']:.2f} ms"
        )
        print(
            f"    avg system routing:   "
            f"{timing.get('avg_system_online_route_decision_ms', 0.0):.2f} ms"
        )
        print(
            f"    simulator key excluded:"
            f"{timing.get('avg_simulator_key_prepare_excluded_ms', 0.0):.2f} ms"
        )
        print(f"    avg query tokenize:   {timing['avg_query_tokenize_ms']:.2f} ms")
        print(f"    avg coarse routing:   {timing['avg_coarse_topic_routing_ms']:.2f} ms")
        print(f"    avg key prepare:      {timing['avg_candidate_key_prepare_ms']:.2f} ms")
        print(f"    avg query-Q prepare:  {timing['avg_query_q_prepare_ms']:.2f} ms")
        print(f"    avg candidate prefilter:{timing['avg_candidate_prefilter_ms']:.2f} ms")
        print(f"    avg dynamic cand pool:{timing['avg_dynamic_candidate_pool_ms']:.2f} ms")
        print(f"    avg Q-K candidates:  {timing['avg_candidates_scored_qk']:.1f}")
        if getattr(args, "adaptive_topic_rescue", False):
            print(
                f"    avg topic rescue trigger/final_k:"
                f"{100 * timing.get('avg_adaptive_topic_rescue_trigger_rate', 0.0):.1f}% / "
                f"{timing.get('avg_adaptive_topic_rescue_final_k', 0.0):.2f}"
            )
            print(
                f"    avg topic gap/top1 ratio:"
                f"{timing.get('avg_adaptive_topic_rescue_gap_ratio', 0.0):.3f} / "
                f"{timing.get('avg_adaptive_topic_rescue_top1_ratio', 0.0):.3f}"
            )
        if getattr(args, "route_adaptive_prefilter", False):
            print(
                f"    avg adaptive prefilter keep/uncert:"
                f"{timing.get('avg_candidate_prefilter_adaptive_keep_ratio', 0.0):.2f} / "
                f"{timing.get('avg_candidate_prefilter_adaptive_uncertainty', 0.0):.2f}"
            )
        print(
            f"    avg dyn pool prune:  "
            f"{100 * timing.get('avg_dynamic_candidate_pool_prune_ratio', 0.0):.1f}%"
        )
        print(f"    avg coarse seg gate:  {timing['avg_coarse_segment_gate_ms']:.2f} ms")
        if getattr(args, "route_adaptive_coarse_segment_gate", False):
            print(
                f"    avg adaptive gate keep/uncert:"
                f"{timing.get('avg_coarse_segment_gate_adaptive_keep_ratio', 0.0):.2f} / "
                f"{timing.get('avg_coarse_segment_gate_adaptive_uncertainty', 0.0):.2f}"
            )
        if str(getattr(args, "node_summary_gate", "none")) != "none":
            print(
                f"    avg node summary gate:"
                f"{timing.get('avg_node_summary_gate_ms', 0.0):.2f} ms "
                f"({timing.get('avg_node_summary_gate_before', 0.0):.1f}"
                f"->{timing.get('avg_node_summary_gate_after', 0.0):.1f}, "
                f"prune {100 * timing.get('avg_node_summary_gate_prune_ratio', 0.0):.1f}%)"
            )
            print(
                f"    avg node sketch prep/score/agg:"
                f"{timing.get('avg_node_summary_prepare_ms', 0.0):.2f} / "
                f"{timing.get('avg_node_summary_score_ms', 0.0):.2f} / "
                f"{timing.get('avg_node_summary_aggregate_ms', 0.0):.2f} ms"
            )
        print(f"    avg Q-K model:        {timing['avg_qk_model_inference_ms']:.2f} ms")
        print(f"    avg Q-K aggregate:    {timing['avg_qk_score_aggregation_ms']:.2f} ms")
        print(f"    avg exact Q-K:        {timing['avg_qk_scoring_ms']:.2f} ms")
        print(f"    avg Q-K total stage:  {timing['avg_qk_total_stage_ms']:.2f} ms")
        print(f"    avg postprocess:      {timing['avg_selection_postprocess_ms']:.2f} ms")
        print(f"    avg unaccounted:      {timing['avg_route_unaccounted_ms']:.2f} ms")
        steady_n = int(timing.get("num_steady_state_timing_cases", 0))
        if steady_n > 0:
            print(
                "    steady-state cases:  "
                f"{steady_n}/{int(timing.get('num_timing_cases', 0))} "
                "(excludes first valid query per doc)"
            )
            print(
                f"    steady online wall:   "
                f"{timing['avg_online_route_decision_ms_steady_state']:.2f} ms"
            )
            print(
                f"    steady system routing:"
                f"{timing.get('avg_system_online_route_decision_ms_steady_state', 0.0):.2f} ms"
            )
            print(f"    steady Q-K model:    {timing['avg_qk_model_inference_ms_steady_state']:.2f} ms")

    system_cost = summary_payload.get("system_cost_summary")
    if system_cost:
        assumptions = system_cost.get("assumptions", {})
        print("\n  Estimated system cost:")
        print(
            f"    model:               "
            f"{assumptions.get('timing_model', 'active_node_v2')} "
            f"({assumptions.get('routing_compute_scope', '')})"
        )
        print(
            f"    assumptions:         "
            f"fetch {float(assumptions.get('fetch_bandwidth_gbps', 0.0)):.1f} Gbps, "
            f"control {float(assumptions.get('control_bandwidth_gbps', 0.0)):.1f} Gbps, "
            f"RTT {float(assumptions.get('per_node_rtt_ms', 0.0)):.2f} ms/node, "
            f"seg {float(assumptions.get('per_segment_overhead_ms', 0.0)):.2f} ms"
        )
        print(
            f"    avg routing wall:    "
            f"{system_cost.get('avg_routing_wall_clock_ms', 0.0):.2f} ms"
        )
        print(
            f"    simulator excluded:  "
            f"{system_cost.get('avg_routing_simulator_excluded_ms', 0.0):.2f} ms"
        )
        if assumptions.get("timing_model") == "active_node_v2":
            print(
                f"    active query dispatch:{system_cost.get('avg_query_dispatch_ms', 0.0):.2f} ms"
            )
            print(
                f"    active summary meta: {system_cost.get('avg_summary_metadata_fetch_ms', 0.0):.2f} ms"
            )
            print(
                f"    active node select:  {system_cost.get('avg_node_local_selection_ms', 0.0):.2f} ms"
            )
            print(
                f"    active cand return:  {system_cost.get('avg_candidate_return_ms', 0.0):.2f} ms"
            )
            print(
                f"    active coord fusion: {system_cost.get('avg_coordinator_fusion_ms', 0.0):.2f} ms"
            )
            print(
                f"    offline summary prep:{system_cost.get('avg_offline_summary_prepare_ms', 0.0):.2f} ms excluded"
            )
            print(
                f"    oracle exact Q-K:    {system_cost.get('avg_oracle_exact_qk_ms_excluded', 0.0):.2f} ms excluded"
            )
        else:
            print(
                f"    avg system routing:  "
                f"{system_cost['avg_routing_overhead_ms']:.2f} ms"
            )
        print(f"    avg selected KV:     {system_cost['avg_selected_kv_mib']:.1f} MiB")
        print(f"    avg full KV:         {system_cost['avg_full_kv_mib']:.1f} MiB")
        print(f"    avg KV reduction:    {100 * system_cost['avg_kv_bytes_reduction_ratio']:.1f}%")
        print(f"    avg selected fetch:  {system_cost['avg_selected_fetch_latency_ms']:.2f} ms")
        print(f"    avg full fetch:      {system_cost['avg_full_fetch_latency_ms']:.2f} ms")
        print(
            f"    avg fetch reduction: "
            f"{100 * system_cost['avg_fetch_latency_reduction_ratio']:.1f}%"
        )
        print(f"    avg first decode:    {system_cost['avg_selected_decode_time_ms']:.2f} ms")
        print(f"    avg selected TTFT:   {system_cost['avg_selected_ttft_ms']:.2f} ms")
        print(f"    avg full TTFT:       {system_cost['avg_full_ttft_ms']:.2f} ms")
        print(f"    avg TTFT reduction:  {100 * system_cost['avg_ttft_reduction_ratio']:.1f}%")
        if assumptions.get("timing_model") == "active_node_v2":
            print(
                f"    legacy selected TTFT:{system_cost.get('avg_legacy_selected_ttft_ms', 0.0):.2f} ms"
            )
            print(
                f"    legacy TTFT reduction:"
                f"{100 * system_cost.get('avg_legacy_ttft_reduction_ratio', 0.0):.1f}%"
            )
        steady_cost = summary_payload.get("system_cost_steady_state_summary")
        if steady_cost:
            print(f"    steady selected TTFT:{steady_cost['avg_selected_ttft_ms']:.2f} ms")
            print(
                f"    steady TTFT reduction:"
                f"{100 * steady_cost['avg_ttft_reduction_ratio']:.1f}%"
            )
    cachegen = summary_payload.get("cachegen_full_summary")
    if cachegen:
        print("\n  CacheGen-full estimated baseline:")
        if cachegen.get("status") != "ok":
            print(f"    status:              error ({cachegen.get('first_error_type', '')})")
            print(f"    first error:         {cachegen.get('first_error', '')}")
        else:
            print("    quality proxy:       full-context answer F1 (no CacheGen decode-answer yet)")
            print(f"    avg compressed KV:   {cachegen['avg_compressed_mib']:.1f} MiB")
            print(
                f"    avg compression save:"
                f"{100 * cachegen['avg_compression_saving_ratio']:.1f}%"
            )
            print(f"    avg encode total:    {cachegen['avg_cachegen_total_encode_ms']:.2f} ms")
            print(f"    avg fetch:           {cachegen['avg_cachegen_fetch_latency_ms']:.2f} ms")
            print(f"    avg TTFT:            {cachegen['avg_cachegen_ttft_ms']:.2f} ms")
            print(
                f"    selected-cachegen TTFT delta: "
                f"{cachegen['avg_selected_vs_cachegen_ttft_delta_ms']:.2f} ms"
            )
            print(
                f"    selected vs cachegen TTFT save: "
                f"{100 * cachegen['avg_selected_vs_cachegen_ttft_saving_ratio']:.1f}%"
            )

    cachegen_rt = summary_payload.get("cachegen_roundtrip_answer_summary")
    if cachegen_rt:
        print("\n  CacheGen-full roundtrip answer baseline:")
        if cachegen_rt.get("status") != "ok":
            print(
                f"    status:              error "
                f"({cachegen_rt.get('first_error_type', '')})"
            )
            print(f"    first error:         {cachegen_rt.get('first_error', '')}")
        else:
            print("    quality:             generated from decompressed CacheGen KV")
            print(f"    avg answer F1:       {100 * cachegen_rt['avg_answer_f1']:.1f}%")
            print(
                f"    avg delta vs full:   "
                f"{100 * cachegen_rt['avg_answer_f1_delta_vs_full']:.1f}%"
            )
            print(
                f"    avg delta vs selected:"
                f"{100 * cachegen_rt['avg_answer_f1_delta_vs_selected']:.1f}%"
            )
            print(
                f"    bad output rate:     "
                f"{100 * cachegen_rt['bad_output_rate']:.1f}%"
            )
            print(f"    avg compressed KV:   {cachegen_rt['avg_compressed_mib']:.1f} MiB")
            print(f"    avg encode total:    {cachegen_rt['avg_total_encode_ms']:.2f} ms")
            print(f"    avg decode measured: {cachegen_rt['avg_decode_ms']:.2f} ms")
            print(f"    avg fetch:           {cachegen_rt['avg_fetch_latency_ms']:.2f} ms")
            print(f"    avg TTFT:            {cachegen_rt['avg_ttft_ms']:.2f} ms")

    answer = summary_payload.get("answer_summary")
    if answer:
        print("\n  Answer quality (full vs selective):")
        print(f"    avg full-answer F1:      {100 * answer['avg_full_answer_f1']:.1f}%")
        print(f"    avg selective-answer F1: {100 * answer['avg_selected_answer_f1']:.1f}%")
        if int(answer.get("num_oracle_answer_cases", 0)) > 0:
            print(f"    avg oracle-answer F1:    {100 * answer['avg_oracle_answer_f1']:.1f}%")
        else:
            print("    avg oracle-answer F1:    skipped")
        print(f"    avg F1 delta:            {100 * answer['avg_answer_f1_delta']:.1f}%")
        if int(answer.get("num_oracle_answer_cases", 0)) > 0:
            print(
                f"    avg oracle-full delta:   "
                f"{100 * answer['avg_oracle_answer_f1_delta_vs_full']:.1f}%"
            )
            print(
                f"    avg selected-oracle gap: "
                f"{100 * answer['avg_selected_answer_f1_delta_vs_oracle']:.1f}%"
            )
        print(f"    avg full context tokens: {answer['avg_full_context_tokens']:.1f}")
        print(f"    avg selected ctx tokens: {answer['avg_selected_context_tokens']:.1f}")
        if int(answer.get("num_oracle_answer_cases", 0)) > 0:
            print(f"    avg oracle ctx tokens:   {answer['avg_oracle_context_tokens']:.1f}")
        print(
            f"    avg ctx token saving:    "
            f"{100 * answer['avg_context_token_saving_ratio']:.1f}%"
        )
        print(f"    selective >= full:       {100 * answer['selected_ge_full_rate']:.1f}%")
        if int(answer.get("num_oracle_answer_cases", 0)) > 0:
            print(f"    oracle >= full:          {100 * answer['oracle_ge_full_rate']:.1f}%")
            print(f"    selective >= oracle:     {100 * answer['selected_ge_oracle_rate']:.1f}%")
            print(
                "    bad output rate full/sel/oracle: "
                f"{100 * answer['full_bad_output_rate']:.1f}% / "
                f"{100 * answer['selected_bad_output_rate']:.1f}% / "
                f"{100 * answer['oracle_bad_output_rate']:.1f}%"
            )
        else:
            print(
                "    bad output rate full/sel: "
                f"{100 * answer['full_bad_output_rate']:.1f}% / "
                f"{100 * answer['selected_bad_output_rate']:.1f}%"
            )
