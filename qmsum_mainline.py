"""
================================================================================
 QMSum 主线实验入口（干净版）
================================================================================

当前主线方案：
  1. lexical (BM25) 粗粒度 topic 路由 → 选 top-1 topic
  2. 只在这个选中的 topic 内部做 Q-K attention 细粒度 chunk 选择
  3. 用 QMSum 标注的 relevant_text_span 验证选中的 turn 是否覆盖了证据
  4. 统计传输成本（chunk 数 / segment 数 / token 数）
  5. 可选：生成答案并比较 full vs selected 的 F1

这个文件只保留当前冻结的主线逻辑，"不会"引入其他策略分支。
所有历史分支 / 多策略对比 / rerank / lexical_prf 等实验留在：
  - qmsum_sim.py      (总控，支持多种 coarse 策略对比)
  - qmsum_routing.py   (通用路由逻辑，包含 embedding / lexical / rerank 等)

如果你只想理解当前在做什么，从这个文件开始读就够。

完整数据流（8 步）：
  1. 从 QMSum 取一个 meeting
  2. 取 meeting 里的一个 specific query
  3. 把整场 meeting transcript 拼成 prompt → 记录每个 turn 的 token 边界
  4. 跑模型 prefill，拿到全量 KV cache
  5. 用 topic_list 建 topic 节点 + lexical 粗选
  6. 只在选中 topic 内部，用 Q-K 分数选 top-k chunks
  7. 用 relevant_text_span 检查选中的 turn 是否覆盖了正确证据
  8. 统计传输成本 + 可选答案生成对比
================================================================================
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

# 确保可以 import 同级目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 各模块职责速查 ----
# qmsum_data:         数据加载 / prompt 拼接 / turn→topic 映射
# qmsum_mainline_routing:  lexical 粗选 + topic 内 Q-K chunk 细选（当前主线核心）
# qmsum_eval:         证据命中评估 + 传输成本统计 + 汇总
# qmsum_answering:    答案生成 + 文本 F1 计算
# qmsum_output:       结果输出（TSV / JSONL / Markdown）
# qmsum_trace:        单 case 详细 trace 导出（调试 / 汇报用）
from qmsum_answering import build_answer_eval
from qmsum_cachegen import (
    build_cachegen_case_metrics,
    build_cachegen_full_estimate,
    build_cachegen_roundtrip_answer_eval,
)
from qmsum_data import build_qmsum_prompt, load_mainline_sample, spans_to_turn_set
from qmsum_eval import (
    build_system_cost_config,
    build_system_cost_estimate,
    build_summary_payload,
    build_transfer_accounting,
    compute_selected_turn_metrics,
    print_summary,
)
from qmsum_mainline_config import (
    add_mainline_profile_argument,
    apply_mainline_profile,
    collect_explicit_arg_dests,
)
from qmsum_mainline_routing import (
    build_mainline_doc_routing_artifacts,
    build_mainline_topic_index_qmsum,
    score_mainline_topic_chunk,
)
from qmsum_output import (
    build_output_suffix,
    write_case_answer_log,
    write_case_answer_markdown,
    write_case_summary_tsv,
)
from qmsum_trace import preview_text, should_trace_case, write_case_trace


def _format_config_value(value):
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _print_config_section(title, items):
    print(f"  [{title}]")
    for name, value in items:
        print(f"    {name}: {_format_config_value(value)}")


def _print_compact_mainline_config(args):
    print("=" * 70)
    print("QMSum mainline simulation")
    _print_config_section(
        "run",
        [
            ("mainline_profile", getattr(args, "applied_mainline_profile", "manual")),
            ("dataset", args.dataset),
            ("data", args.data_path),
            ("docs", f"{args.start_doc}:{args.end_doc}"),
            ("max_queries_per_doc", args.max_queries_per_doc),
            ("num_nodes", args.num_nodes),
            ("seed", args.seed),
        ],
    )
    _print_config_section(
        "model",
        [
            ("model_id", args.model_id),
            ("model_loader", args.model_loader),
            ("hf_quantization", args.hf_quantization),
            ("hf_dtype", args.hf_dtype),
            ("hf_attn_impl", args.hf_attn_impl),
            ("hf_device_map", args.hf_device_map),
        ],
    )
    _print_config_section(
        "routing",
        [
            ("routing_granularity", "hierarchical"),
            ("coarse_strategy", "lexical"),
            ("hier_top_topics", args.hier_top_topics),
            ("candidate_topic_scope", args.candidate_topic_scope),
            ("route_selection_mode", args.route_selection_mode),
            ("route_chunk_size", args.route_chunk_size),
            ("route_top_k", args.route_top_k),
            ("route_per_head", args.route_per_head),
            ("route_candidate_prefilter", args.route_candidate_prefilter),
            ("route_coarse_segment_gate", args.route_coarse_segment_gate),
            ("node_summary_gate", args.node_summary_gate),
            ("node_summary_gate_summary_mode", args.node_summary_gate_summary_mode),
            ("node_summary_gate_budget_mode", args.node_summary_gate_budget_mode),
        ],
    )
    _print_config_section(
        "qk",
        [
            ("qk_aggregation", args.qk_aggregation),
            ("qk_topk", args.qk_topk),
            ("qk_token_pooling", args.qk_token_pooling),
            ("qk_query_topk_ratio", args.qk_query_topk_ratio),
            ("qk_score_batch_size", args.qk_score_batch_size),
            ("cache_candidate_keys", args.cache_candidate_keys),
            ("cache_query_q", args.cache_query_q),
        ],
    )
    _print_config_section(
        "cost_output",
        [
            ("ttft_model", args.ttft_model),
            ("fetch_bandwidth_gbps", args.fetch_bandwidth_gbps),
            ("control_bandwidth_gbps", args.control_bandwidth_gbps),
            ("per_node_rtt_ms", args.per_node_rtt_ms),
            ("control_rtt_ms", args.control_rtt_ms),
            ("per_segment_overhead_ms", args.per_segment_overhead_ms),
            ("per_rpc_overhead_ms", args.per_rpc_overhead_ms),
            ("decode_startup_ms", args.decode_startup_ms),
            ("kv_cache_dtype_bytes", args.kv_cache_dtype_bytes),
            ("eval_answers", args.eval_answers),
            ("light_output", args.light_output),
            ("case_summary_mode", getattr(args, "case_summary_mode", "compact")),
        ],
    )
    if args.eval_cachegen_full or args.eval_cachegen_roundtrip_answer:
        _print_config_section(
            "cachegen",
            [
                ("eval_cachegen_full", args.eval_cachegen_full),
                ("eval_cachegen_roundtrip_answer", args.eval_cachegen_roundtrip_answer),
                ("cachegen_model_name", args.cachegen_model_name),
                ("cachegen_quant_level", args.cachegen_quant_level),
                ("cachegen_chunk_size", args.cachegen_chunk_size),
                ("cachegen_include_encode_time", args.cachegen_include_encode_time),
                ("cachegen_decode_ms", args.cachegen_decode_ms),
                ("cachegen_segment_count_mode", args.cachegen_segment_count_mode),
            ],
        )
    print("  full config: pass --verbose_config")
    print("=" * 70)


def _print_verbose_mainline_config(args):
    print("=" * 70)
    print("QMSum mainline simulation")
    print("  config_view: verbose")
    for name in sorted(vars(args)):
        print(f"  {name}: {_format_config_value(getattr(args, name))}")
    print("  routing_granularity: hierarchical")
    print("  coarse_strategy: lexical")
    print("=" * 70)


def print_mainline_config(args):
    if getattr(args, "verbose_config", False):
        _print_verbose_mainline_config(args)
    else:
        _print_compact_mainline_config(args)


def main(args):
    # ------------------------------------------------------------------
    # 延迟导入：避免循环依赖，同时让 _chunk_qk_scores_per_head 和
    # split_kv 只在真正跑实验时才加载
    # ------------------------------------------------------------------
    from experiment_chunk_split import _chunk_qk_scores_per_head
    from src.utils import define_model_and_tokenizer, split_kv

    print_mainline_config(args)

    print("\nLoading model...")
    model, tokenizer = define_model_and_tokenizer(
        args.model_id,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        model_loader=args.model_loader,
        hf_quantization=args.hf_quantization,
        hf_dtype=args.hf_dtype,
        hf_attn_impl=args.hf_attn_impl,
        hf_device_map=args.hf_device_map,
    )
    print("Model loaded\n")
    system_cost_config = build_system_cost_config(model.config, args)
    print(
        "System cost model: "
        f"model={system_cost_config['ttft_model']}, "
        f"kv_bytes/token={system_cost_config['kv_bytes_per_token']:.0f}, "
        f"fetch_bw={system_cost_config['fetch_bandwidth_gbps']:.1f} Gbps, "
        f"control_bw={system_cost_config['control_bandwidth_gbps']:.1f} Gbps\n"
    )

    if args.query_tokenizer_warmup > 0:
        warmup_start_time = time.perf_counter()
        for _ in range(int(args.query_tokenizer_warmup)):
            warmup_ids = tokenizer(
                "Warm up query tokenization for routing timing.",
                return_tensors="pt",
            ).input_ids.cuda()
            del warmup_ids
        torch.cuda.synchronize()
        warmup_ms = 1000.0 * (time.perf_counter() - warmup_start_time)
        print(
            "Query tokenizer warmup: "
            f"{int(args.query_tokenizer_warmup)} pass(es), {warmup_ms:.2f} ms\n"
        )

    # ---- 确定 Q-K 打分的层 ----
    # 默认用 5 层：浅层(0) + 中层(8,16,24) + 最后一层
    # 负索引支持：-1 = 最后一层
    num_model_layers = len(model.model.layers)
    if args.scoring_layers:
        scoring_layers = [int(x) for x in args.scoring_layers.split(",")]
    else:
        scoring_layers = [0, 8, 16, 24, num_model_layers - 1]
    scoring_layers = [l if l >= 0 else num_model_layers + l for l in scoring_layers]

    results = []               # 所有 (doc, query) 的结果条目
    written_trace_paths = []   # 记录写出的 trace 文件路径

    def format_topic_id(topic_id):
        """将 topic id 格式化为 T0, T1, ... 用于打印"""
        return f"T{int(topic_id)}"

    def build_selected_answer_evidence(selected_chunk_details, qk_route_info, mode):
        """Build compact evidence entries for selected answer generation."""
        if mode == "chunks":
            entries = []
            for chunk in selected_chunk_details:
                turn_text = chunk.get("turn_text", "")
                speaker = turn_text.split(":", 1)[0].strip() if ":" in turn_text else "Speaker"
                entries.append(
                    {
                        "turn_idx": int(chunk.get("turn_idx", -1)),
                        "speaker": speaker or "Speaker",
                        "text": chunk.get("chunk_text", ""),
                        "score": float(chunk.get("score", 0.0)),
                    }
                )
            return entries

        if mode != "chunk_turns":
            return []

        by_turn = {}
        for chunk in selected_chunk_details:
            turn_idx = int(chunk.get("turn_idx", -1))
            if turn_idx in by_turn:
                continue
            turn_text = chunk.get("turn_text", "")
            if ":" in turn_text:
                speaker, text = turn_text.split(":", 1)
            else:
                speaker, text = "Speaker", turn_text
            by_turn[turn_idx] = {
                "turn_idx": turn_idx,
                "speaker": speaker.strip() or "Speaker",
                "text": text.strip(),
                "score": float(chunk.get("score", 0.0)),
            }

        ordered_turns = qk_route_info.get("ordered_answer_turns", [])
        ordered_turn_set = set(int(x) for x in ordered_turns)
        ordered_entries = [
            by_turn[int(turn_idx)]
            for turn_idx in ordered_turns
            if int(turn_idx) in by_turn
        ]
        remaining = [
            entry
            for turn_idx, entry in sorted(by_turn.items())
            if turn_idx not in ordered_turn_set
        ]
        return ordered_entries + remaining

    def normalize_qk_route_info(qk_route_info, selected_chunk_details):
        """Fill backward-compatible fields so outputs stay interpretable."""
        qk_route_info = dict(qk_route_info or {})
        routing_timing_breakdown = dict(
            qk_route_info.get("routing_timing_breakdown_ms") or {}
        )

        mode = qk_route_info.get("candidate_prefilter_mode")
        if not mode:
            mode = str(getattr(args, "route_candidate_prefilter", "none")).lower()
            qk_route_info["candidate_prefilter_mode"] = mode

        pool_size = qk_route_info.get("candidate_prefilter_pool_size")
        before_count = qk_route_info.get("num_candidates_before_prefilter")
        after_count = qk_route_info.get("num_candidates_after_prefilter")
        scored_count = qk_route_info.get("num_candidates_scored_qk")
        selected_count = qk_route_info.get("num_selected_candidates")

        if pool_size in [None, ""]:
            if mode == "lexical" and after_count not in [None, ""]:
                pool_size = after_count
            elif mode in ["none", ""]:
                pool_size = before_count
            qk_route_info["candidate_prefilter_pool_size"] = pool_size

        if after_count in [None, ""] and scored_count not in [None, ""]:
            qk_route_info["num_candidates_after_prefilter"] = scored_count

        if before_count in [None, ""]:
            fallback_before = qk_route_info.get("num_candidates_after_topic_filter")
            if fallback_before not in [None, ""]:
                qk_route_info["num_candidates_before_prefilter"] = fallback_before

        preview = list(qk_route_info.get("candidate_prefilter_preview") or [])
        if not preview and mode == "lexical":
            preview = []
            for chunk in selected_chunk_details[:20]:
                score = float(chunk.get("prefilter_score", -1e9))
                if score <= -1e8:
                    continue
                preview.append(
                    {
                        "candidate_id": int(chunk.get("candidate_id", -1)),
                        "score": score,
                    }
                )
            qk_route_info["candidate_prefilter_preview"] = preview

        valid_prefilter_scores = [
            float(chunk.get("prefilter_score", -1e9))
            for chunk in selected_chunk_details
            if float(chunk.get("prefilter_score", -1e9)) > -1e8
        ]
        qk_route_info["prefilter_effective"] = bool(valid_prefilter_scores)
        qk_route_info["prefilter_num_scored_selected_chunks"] = len(valid_prefilter_scores)
        qk_route_info["routing_timing_breakdown_ms"] = routing_timing_breakdown

        if selected_count in [None, ""]:
            qk_route_info["num_selected_candidates"] = len(selected_chunk_details)

        return qk_route_info

    def build_relevant_candidate_survival(route_candidates, relevant_turns, qk_route_info):
        """Trace where gold-turn candidates survive or disappear in the route funnel."""
        relevant_turn_set = {int(turn_idx) for turn_idx in relevant_turns}
        route_candidates = list(route_candidates or [])
        candidate_by_id = {
            int(cand.get("candidate_id", -1)): cand
            for cand in route_candidates
            if int(cand.get("candidate_id", -1)) >= 0
        }
        relevant_candidate_ids = sorted(
            int(candidate_id)
            for candidate_id, cand in candidate_by_id.items()
            if int(cand.get("turn_idx", -1)) in relevant_turn_set
        )

        qk_ranked = list(qk_route_info.get("qk_ranked_candidates") or [])
        qk_rank_by_id = {
            int(item.get("candidate_id", -1)): rank
            for rank, item in enumerate(qk_ranked, start=1)
            if int(item.get("candidate_id", -1)) >= 0
        }
        qk_score_by_id = {
            int(item.get("candidate_id", -1)): float(item.get("score", 0.0))
            for item in qk_ranked
            if int(item.get("candidate_id", -1)) >= 0
        }
        selected_candidate_id_set = {
            int(candidate_id)
            for candidate_id in qk_route_info.get("selected_candidate_ids", [])
        }

        stage_specs = [
            ("candidate_build", sorted(candidate_by_id.keys())),
            (
                "topic_filter",
                qk_route_info.get("candidate_ids_after_topic_filter", []),
            ),
            (
                "candidate_prefilter",
                qk_route_info.get("candidate_ids_after_prefilter", []),
            ),
            (
                "dynamic_candidate_pool",
                qk_route_info.get("candidate_ids_after_dynamic_pool", []),
            ),
            (
                "coarse_segment_gate",
                qk_route_info.get("candidate_ids_after_coarse_segment_gate", []),
            ),
            (
                "node_summary_gate",
                qk_route_info.get("candidate_ids_after_node_summary_gate", []),
            ),
            ("qk_scored", qk_route_info.get("candidate_ids_scored_qk", [])),
            ("qk_selected", qk_route_info.get("selected_candidate_ids", [])),
        ]

        stage_records = []
        stage_turn_sets = {}
        total_relevant_candidates = len(relevant_candidate_ids)
        total_relevant_turns = len(relevant_turn_set)
        previous_turn_count = total_relevant_turns
        first_drop_stage = ""
        first_zero_stage = ""
        for stage_name, raw_ids in stage_specs:
            stage_ids = {int(candidate_id) for candidate_id in (raw_ids or [])}
            kept_relevant_candidate_ids = [
                candidate_id
                for candidate_id in relevant_candidate_ids
                if candidate_id in stage_ids
            ]
            kept_relevant_turns = sorted(
                {
                    int(candidate_by_id[candidate_id].get("turn_idx", -1))
                    for candidate_id in kept_relevant_candidate_ids
                    if candidate_id in candidate_by_id
                }
            )
            stage_turn_sets[stage_name] = set(kept_relevant_turns)
            kept_turn_count = len(kept_relevant_turns)
            if not first_drop_stage and kept_turn_count < previous_turn_count:
                first_drop_stage = stage_name
            if not first_zero_stage and total_relevant_turns > 0 and kept_turn_count == 0:
                first_zero_stage = stage_name
            previous_turn_count = kept_turn_count
            stage_records.append(
                {
                    "stage": stage_name,
                    "candidate_count": len(stage_ids),
                    "relevant_candidate_count": len(kept_relevant_candidate_ids),
                    "relevant_candidate_recall": (
                        float(len(kept_relevant_candidate_ids))
                        / max(1.0, float(total_relevant_candidates))
                    ),
                    "relevant_turn_count": kept_turn_count,
                    "relevant_turn_recall": (
                        float(kept_turn_count) / max(1.0, float(total_relevant_turns))
                    ),
                    "relevant_turns": kept_relevant_turns[:32],
                }
            )

        selected_turn_recall = (
            float(len(stage_turn_sets.get("qk_selected", set())))
            / max(1.0, float(total_relevant_turns))
        )
        if selected_turn_recall > 0.0:
            failure_stage = "partial_or_ok"
        elif first_zero_stage:
            failure_stage = first_zero_stage
        else:
            failure_stage = "unknown"

        relevant_turn_preview = []
        for turn_idx in sorted(relevant_turn_set)[:24]:
            turn_candidate_ids = [
                candidate_id
                for candidate_id in relevant_candidate_ids
                if int(candidate_by_id[candidate_id].get("turn_idx", -1)) == int(turn_idx)
            ]
            scored_ids = [cid for cid in turn_candidate_ids if cid in qk_rank_by_id]
            selected_ids = [
                cid for cid in turn_candidate_ids if cid in selected_candidate_id_set
            ]
            best_rank = min((qk_rank_by_id[cid] for cid in scored_ids), default=0)
            best_score = (
                max((qk_score_by_id[cid] for cid in scored_ids), default=0.0)
                if scored_ids
                else 0.0
            )
            relevant_turn_preview.append(
                {
                    "turn_idx": int(turn_idx),
                    "candidate_ids": turn_candidate_ids,
                    "survived_stages": [
                        stage_name
                        for stage_name, _ in stage_specs
                        if int(turn_idx) in stage_turn_sets.get(stage_name, set())
                    ],
                    "best_qk_rank": int(best_rank),
                    "best_qk_score": float(best_score),
                    "selected_candidate_ids": selected_ids,
                }
            )

        stage_by_name = {item["stage"]: item for item in stage_records}
        selected_budget = len(selected_candidate_id_set)
        ranked_top_budget = qk_ranked[:selected_budget] if selected_budget > 0 else []
        ranked_top_budget_turns = sorted(
            {
                int(item.get("turn_idx", -1))
                for item in ranked_top_budget
                if int(item.get("turn_idx", -1)) in relevant_turn_set
            }
        )

        best_rank_by_turn = {}
        best_candidate_by_turn = {}
        for rank, item in enumerate(qk_ranked, start=1):
            turn_idx = int(item.get("turn_idx", -1))
            if turn_idx not in relevant_turn_set or turn_idx in best_rank_by_turn:
                continue
            best_rank_by_turn[turn_idx] = int(rank)
            best_candidate_by_turn[turn_idx] = {
                "candidate_id": int(item.get("candidate_id", -1)),
                "turn_idx": int(turn_idx),
                "qk_rank": int(rank),
                "score": float(item.get("score", 0.0)),
            }

        scored_relevant_turns = sorted(best_rank_by_turn.keys())
        oracle_turn_items = sorted(
            best_candidate_by_turn.values(),
            key=lambda item: (int(item["qk_rank"]), int(item["turn_idx"])),
        )
        oracle_turn_items_at_budget = (
            oracle_turn_items[:selected_budget] if selected_budget > 0 else []
        )
        oracle_turns_at_budget = [
            int(item["turn_idx"]) for item in oracle_turn_items_at_budget
        ]
        qk_selected_turn_recall = float(
            stage_by_name.get("qk_selected", {}).get("relevant_turn_recall", 0.0)
        )
        qk_scored_turn_recall = float(
            stage_by_name.get("qk_scored", {}).get("relevant_turn_recall", 0.0)
        )
        qk_ranked_top_budget_turn_recall = (
            float(len(ranked_top_budget_turns)) / max(1.0, float(total_relevant_turns))
        )
        qk_oracle_turn_recall_at_selected_budget = (
            float(len(oracle_turns_at_budget)) / max(1.0, float(total_relevant_turns))
        )
        qk_selector_gap = max(
            0.0,
            qk_oracle_turn_recall_at_selected_budget - qk_selected_turn_recall,
        )
        qk_budget_cap_gap = max(
            0.0,
            qk_scored_turn_recall - qk_oracle_turn_recall_at_selected_budget,
        )
        best_ranks = sorted(int(x) for x in best_rank_by_turn.values())
        first_relevant_rank = best_ranks[0] if best_ranks else 0
        num_qk_ranked = len(qk_ranked)
        first_relevant_rank_pct = (
            float(first_relevant_rank) / float(num_qk_ranked)
            if first_relevant_rank and num_qk_ranked
            else 0.0
        )
        if best_ranks:
            mean_best_rank = float(sum(best_ranks)) / float(len(best_ranks))
            mid = len(best_ranks) // 2
            if len(best_ranks) % 2:
                median_best_rank = float(best_ranks[mid])
            else:
                median_best_rank = float(best_ranks[mid - 1] + best_ranks[mid]) / 2.0
        else:
            mean_best_rank = 0.0
            median_best_rank = 0.0

        selection_debug = qk_route_info.get("route_selection_debug") or {}
        turn_utility_ranked = list(
            selection_debug.get("turn_utility_ranked_turns") or []
        )
        turn_utility_rank_by_turn = {}
        turn_utility_gold_items = []
        for item in turn_utility_ranked:
            turn_idx = int(item.get("turn_idx", -1))
            rank = int(item.get("rank", 0))
            if turn_idx < 0 or rank <= 0 or turn_idx in turn_utility_rank_by_turn:
                continue
            turn_utility_rank_by_turn[turn_idx] = rank
            if turn_idx in relevant_turn_set:
                turn_utility_gold_items.append(
                    {
                        "turn_idx": int(turn_idx),
                        "rank": int(rank),
                        "best_candidate_id": int(item.get("best_candidate_id", -1)),
                        "utility": float(item.get("utility", 0.0)),
                        "best_qk": float(item.get("best_qk", 0.0)),
                        "head_vote": float(item.get("head_vote", 0.0)),
                        "prefilter_score": float(item.get("prefilter_score", 0.0)),
                        "num_scored_chunks": int(item.get("num_scored_chunks", 0)),
                    }
                )
        turn_utility_gold_items = sorted(
            turn_utility_gold_items,
            key=lambda item: (int(item["rank"]), int(item["turn_idx"])),
        )
        turn_utility_top_budget = (
            turn_utility_ranked[:selected_budget] if selected_budget > 0 else []
        )
        turn_utility_top_budget_turns = sorted(
            {
                int(item.get("turn_idx", -1))
                for item in turn_utility_top_budget
                if int(item.get("turn_idx", -1)) in relevant_turn_set
            }
        )
        turn_utility_recall_at_selected_budget = (
            float(len(turn_utility_top_budget_turns))
            / max(1.0, float(total_relevant_turns))
        )
        turn_utility_gold_ranks = [
            int(item["rank"]) for item in turn_utility_gold_items
        ]
        turn_utility_first_relevant_rank = (
            min(turn_utility_gold_ranks) if turn_utility_gold_ranks else 0
        )
        num_turn_utility_ranked = len(turn_utility_ranked)
        turn_utility_first_relevant_rank_pct = (
            float(turn_utility_first_relevant_rank)
            / float(num_turn_utility_ranked)
            if turn_utility_first_relevant_rank and num_turn_utility_ranked
            else 0.0
        )
        if turn_utility_gold_ranks:
            turn_utility_mean_gold_rank = (
                float(sum(turn_utility_gold_ranks))
                / float(len(turn_utility_gold_ranks))
            )
        else:
            turn_utility_mean_gold_rank = 0.0
        turn_utility_gap_to_qk_oracle = max(
            0.0,
            qk_oracle_turn_recall_at_selected_budget
            - turn_utility_recall_at_selected_budget,
        )
        turn_utility_gap_to_scored = max(
            0.0,
            qk_scored_turn_recall - turn_utility_recall_at_selected_budget,
        )

        if total_relevant_turns <= 0:
            qk_oracle_diagnosis = "no_relevant_turns"
        elif qk_scored_turn_recall <= 0.0:
            qk_oracle_diagnosis = "upstream_no_scored_gold"
        elif qk_selector_gap > 1e-9:
            qk_oracle_diagnosis = "selection_gap_after_scoring"
        elif qk_budget_cap_gap > 1e-9:
            qk_oracle_diagnosis = "budget_cap_after_scoring"
        else:
            qk_oracle_diagnosis = "selection_matches_oracle"

        return {
            "total_relevant_turns": int(total_relevant_turns),
            "total_relevant_candidates": int(total_relevant_candidates),
            "first_drop_stage": first_drop_stage or "none",
            "first_zero_stage": first_zero_stage or "none",
            "failure_stage": failure_stage,
            "candidate_build_turn_recall": float(
                stage_by_name.get("candidate_build", {}).get("relevant_turn_recall", 0.0)
            ),
            "topic_filter_turn_recall": float(
                stage_by_name.get("topic_filter", {}).get("relevant_turn_recall", 0.0)
            ),
            "candidate_prefilter_turn_recall": float(
                stage_by_name.get("candidate_prefilter", {}).get("relevant_turn_recall", 0.0)
            ),
            "dynamic_candidate_pool_turn_recall": float(
                stage_by_name.get("dynamic_candidate_pool", {}).get("relevant_turn_recall", 0.0)
            ),
            "coarse_segment_gate_turn_recall": float(
                stage_by_name.get("coarse_segment_gate", {}).get("relevant_turn_recall", 0.0)
            ),
            "node_summary_gate_turn_recall": float(
                stage_by_name.get("node_summary_gate", {}).get("relevant_turn_recall", 0.0)
            ),
            "qk_scored_turn_recall": float(
                stage_by_name.get("qk_scored", {}).get("relevant_turn_recall", 0.0)
            ),
            "qk_selected_turn_recall": float(
                stage_by_name.get("qk_selected", {}).get("relevant_turn_recall", 0.0)
            ),
            "qk_selection_budget": int(selected_budget),
            "qk_ranked_candidate_count": int(num_qk_ranked),
            "qk_scored_relevant_turn_count": int(len(scored_relevant_turns)),
            "qk_ranked_top_budget_turn_recall": float(
                qk_ranked_top_budget_turn_recall
            ),
            "qk_oracle_turn_hit_at_selected_budget": bool(oracle_turns_at_budget),
            "qk_oracle_turn_recall_at_selected_budget": float(
                qk_oracle_turn_recall_at_selected_budget
            ),
            "qk_selector_gap": float(qk_selector_gap),
            "qk_budget_cap_gap": float(qk_budget_cap_gap),
            "qk_first_relevant_rank": int(first_relevant_rank),
            "qk_first_relevant_rank_pct": float(first_relevant_rank_pct),
            "qk_relevant_turn_best_rank_mean": float(mean_best_rank),
            "qk_relevant_turn_best_rank_median": float(median_best_rank),
            "qk_oracle_diagnosis": qk_oracle_diagnosis,
            "turn_utility_ranked_turn_count": int(num_turn_utility_ranked),
            "turn_utility_first_relevant_rank": int(
                turn_utility_first_relevant_rank
            ),
            "turn_utility_first_relevant_rank_pct": float(
                turn_utility_first_relevant_rank_pct
            ),
            "turn_utility_mean_gold_rank": float(turn_utility_mean_gold_rank),
            "turn_utility_recall_at_selected_budget": float(
                turn_utility_recall_at_selected_budget
            ),
            "turn_utility_gap_to_qk_oracle": float(turn_utility_gap_to_qk_oracle),
            "turn_utility_gap_to_scored": float(turn_utility_gap_to_scored),
            "turn_utility_top_budget_turns": turn_utility_top_budget_turns[:32],
            "turn_utility_gold_preview": turn_utility_gold_items[:20],
            "qk_ranked_top_budget_turns": ranked_top_budget_turns[:32],
            "qk_oracle_turns_at_selected_budget": oracle_turns_at_budget[:32],
            "qk_oracle_candidate_preview": oracle_turn_items_at_budget[:20],
            "stage_records": stage_records,
            "relevant_turn_preview": relevant_turn_preview,
        }

    # ==================================================================
    # 主循环：遍历每个 doc 的每个 query
    # ==================================================================
    for doc_id in range(args.start_doc, args.end_doc):
        # --------------------------------------------------------------
        # Phase 2: 加载一个 meeting 样本
        # meeting 结构：{ meeting_id, meeting_transcripts, topic_list, specific_query_list }
        # --------------------------------------------------------------
        try:
            meeting = load_mainline_sample(args.data_path, doc_id=doc_id, dataset=args.dataset)
        except IndexError:
            break

        transcripts = meeting.get("meeting_transcripts", [])
        if len(transcripts) < 4:
            continue  # 太短的会议跳过

        # 限制每个 doc 的 query 数量
        specific_queries = meeting.get("specific_query_list", [])
        if args.max_queries_per_doc > 0:
            specific_queries = specific_queries[: args.max_queries_per_doc]

        # --------------------------------------------------------------
        # Phase 3: 构建 topic 索引（从 QMSum 标注的 topic_list）
        # topic_index 包含：
        #   - topic_nodes: 每个 topic 的 id / label / turns
        #   - turn_to_topic_ids: 每个 turn 属于哪些 topic
        #   - topic_repr_turns: 每个 topic 的代表性 turn（用于 lexical 表示）
        #   - topic_repr_entries: 每个 topic 的文本表示条目
        # --------------------------------------------------------------
        topic_index = build_mainline_topic_index_qmsum(
            meeting,
            transcripts,
            topic_prototype_turns=args.topic_prototype_turns,
            topic_representation_template=args.topic_representation_template,
        )

        # --------------------------------------------------------------
        # Phase 4: 拼接 prompt + 做 prefill 拿到全量 KV cache
        # prompt_text: 所有 turn 拼接成的长文本
        # turn_boundaries: 每个 turn 的 (start_token, end_token)
        # kv_3d: tuple of (K, V) per layer, 每个 shape (num_heads, seq_len, head_dim)
        # --------------------------------------------------------------
        prompt_text, turn_boundaries = build_qmsum_prompt(transcripts, tokenizer)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs.input_ids.cuda()
        total_tokens = input_ids.shape[1]

        # 超长文档保护：如果 token 数超过 max_tokens 则跳过
        if args.max_tokens > 0 and total_tokens > args.max_tokens:
            print(f"  [SKIP] doc_id={doc_id}: {total_tokens} tokens > {args.max_tokens}")
            del input_ids
            torch.cuda.empty_cache()
            continue

        # 前向传播拿到 KV cache（不用 generate，只 prefill）
        try:
            with torch.no_grad():
                outputs = model.model(input_ids=input_ids, use_cache=True)
            torch.cuda.synchronize()
            # squeeze batch dim: (1, heads, seq, dim) → (heads, seq, dim)
            kv_3d = tuple((layer[0][0], layer[1][0]) for layer in outputs.past_key_values)
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM SKIP] doc_id={doc_id}: {total_tokens} tokens")
            del input_ids
            torch.cuda.empty_cache()
            continue

        cachegen_doc_estimate = None
        if args.eval_cachegen_full:
            print(f"  CacheGen full-KV estimate for doc_id={doc_id}...")
            cachegen_doc_estimate = build_cachegen_full_estimate(
                kv_3d,
                total_tokens=total_tokens,
                cost_config=system_cost_config,
                cachegen_model_name=args.cachegen_model_name,
                quant_level=args.cachegen_quant_level,
                chunk_size=args.cachegen_chunk_size,
                include_encode_time=bool(args.cachegen_include_encode_time),
                cachegen_decode_ms=args.cachegen_decode_ms,
                segment_count_mode=args.cachegen_segment_count_mode,
            )
            if cachegen_doc_estimate.get("status") == "ok":
                print(
                    "    CacheGen full: "
                    f"{cachegen_doc_estimate['compressed_mib']:.1f} MiB, "
                    f"encode={cachegen_doc_estimate['total_encode_ms']:.1f} ms, "
                    f"fetch={cachegen_doc_estimate['fetch_latency_ms']:.1f} ms, "
                    f"ttft={cachegen_doc_estimate['estimated_ttft_ms']:.1f} ms"
                )
            else:
                print(
                    "    [CacheGen SKIP] "
                    f"{cachegen_doc_estimate.get('error_type', 'error')}: "
                    f"{cachegen_doc_estimate.get('error', '')}"
                )

        offline_route_artifact_start = time.perf_counter()
        routing_artifacts = build_mainline_doc_routing_artifacts(
            meeting,
            transcripts,
            turn_boundaries,
            total_tokens,
            args,
            topic_index=topic_index,
        )
        offline_route_artifact_prep_ms = 1000.0 * (
            time.perf_counter() - offline_route_artifact_start
        )

        # ==========================================================
        # 对每个 query 做路由 + 评测
        # ==========================================================
        first_timed_query_for_doc = True
        for query_idx, query_info in enumerate(specific_queries):
            # ---- 提取 query 文本和 GT 相关 turn ----
            query_text = query_info.get("query", "").strip()
            # relevant_text_span: QMSum 标注的答案相关区间
            # spans_to_turn_set: 将 span [(start, end), ...] → turn 编号集合
            relevant_turns = spans_to_turn_set(query_info.get("relevant_text_span", []))
            if not query_text or not relevant_turns:
                continue

            timing_is_first_query = bool(first_timed_query_for_doc)
            first_timed_query_for_doc = False

            routing_start_time = time.perf_counter()

            # tokenize query（Q-K 打分用）
            query_tokenize_start_time = time.perf_counter()
            query_ids = tokenizer(query_text, return_tensors="pt").input_ids.cuda()
            query_tokenize_ms = 1000.0 * (time.perf_counter() - query_tokenize_start_time)

            # ======================================================
            # Phase 5: 两级路由 (coarse topic → fine chunk)
            #
            # score_mainline_topic_chunk 做了：
            #   1. lexical BM25 粗选 → 选 top-1 topic
            #   2. 只在选中 topic 内部 build chunk candidates
            #   3. Q-K attention 对每个 candidate chunk 打分
            #   4. 选 top-k chunks（支持 per_head / neighbor_expand）
            #
            # 返回值：
            #   topic_nodes:       所有 topic 节点列表
            #   topic_turn_info:   {topic_id: [turn_idx, ...]}
            #   turn_to_topic_ids: [[topic_id, ...] per turn]
            #   topic_scores:      {topic_id: Q-K score}
            #   strategy_results:  {lexical: {...}, qk: {...}} 排名结果
            #   qk_route_info:     详细路由元信息
            #   selected_candidates: 最终选中的 chunk 列表
            # ======================================================
            (
                topic_nodes,
                topic_turn_info,
                turn_to_topic_ids,
                topic_scores,
                strategy_results,
                qk_route_info,
                selected_candidates,
            ) = score_mainline_topic_chunk(
                kv_3d,
                turn_boundaries,
                total_tokens,
                query_text,
                query_ids,
                model,
                scoring_layers,
                split_kv,
                _chunk_qk_scores_per_head,
                meeting,
                transcripts,
                args,
                topic_index=topic_index,
                routing_artifacts=routing_artifacts,
            )
            routing_overhead_ms = 1000.0 * (time.perf_counter() - routing_start_time)
            routing_breakdown = dict(qk_route_info.get("routing_timing_breakdown_ms", {}) or {})
            simulator_key_prepare_ms = float(
                routing_breakdown.get("candidate_key_prepare_ms", 0.0)
                + routing_breakdown.get("node_summary_prepare_ms", 0.0)
            )
            system_routing_overhead_ms = max(
                0.0,
                float(routing_overhead_ms) - simulator_key_prepare_ms,
            )
            known_routing_component_ms = (
                float(query_tokenize_ms)
                + float(routing_breakdown.get("coarse_topic_routing_ms", 0.0))
                + float(routing_breakdown.get("topic_filter_ms", 0.0))
                + float(routing_breakdown.get("candidate_prefilter_ms", 0.0))
                + float(routing_breakdown.get("dynamic_candidate_pool_ms", 0.0))
                + float(routing_breakdown.get("coarse_segment_gate_ms", 0.0))
                + float(routing_breakdown.get("node_summary_gate_ms", 0.0))
                + float(routing_breakdown.get("candidate_key_prepare_ms", 0.0))
                + float(routing_breakdown.get("qk_scoring_ms", 0.0))
                + float(routing_breakdown.get("selection_postprocess_ms", 0.0))
            )
            route_unaccounted_ms = max(0.0, routing_overhead_ms - known_routing_component_ms)

            # ---- 统计 GT 相关 topic ----
            # 将 relevant_turns 映射到它们所属的 topic
            relevant_topic_counts = defaultdict(int)
            for turn_idx in relevant_turns:
                for topic_id in turn_to_topic_ids[turn_idx]:
                    relevant_topic_counts[int(topic_id)] += 1
            relevant_units = sorted(relevant_topic_counts.keys())
            # dominant_unit: 包含最多 relevant turns 的那个 topic
            dominant_unit = max(
                sorted(relevant_topic_counts.items()),
                key=lambda x: (x[1], -x[0]),
            )[0]

            # ---- Q-K 分数的区分度统计 ----
            qk_values = [s for s in topic_scores.values() if s > -1e8]
            score_variance = (
                float(torch.tensor(qk_values).var().item()) if len(qk_values) > 1 else 0.0
            )
            score_range = float(max(qk_values) - min(qk_values)) if qk_values else 0.0

            # ======================================================
            # Phase 6: 评测
            # ======================================================

            # 6a. Turn 级证据命中率
            #     选中的 chunk 覆盖了多少 GT relevant turn
            selected_turn_metrics = compute_selected_turn_metrics(
                selected_candidates,
                relevant_turns,
            )

            # 6b. 传输成本统计
            #     如果真的要传输这些 KV chunk：
            #     - 涉及多少 topic（transfer unit）
            #     - 连续 chunk 可以合并成多少 segment
            #     - coalescing gain：segment 合并带来的节省
            transfer_accounting = build_transfer_accounting(
                selected_candidates,
                transfer_unit_type="topics",
                unit_field="transfer_topic_id",
            )
            virtual_node_transfer_accounting = build_transfer_accounting(
                selected_candidates,
                transfer_unit_type="virtual_nodes",
                unit_field="transfer_node_id",
            )
            qk_route_info["transfer_accounting"] = transfer_accounting
            qk_route_info["virtual_node_transfer_accounting"] = (
                virtual_node_transfer_accounting
            )
            system_cost = build_system_cost_estimate(
                total_tokens=total_tokens,
                num_virtual_nodes=len(qk_route_info.get("virtual_node_layout", [])),
                transfer_accounting=virtual_node_transfer_accounting,
                routing_overhead_ms=system_routing_overhead_ms,
                cost_config=system_cost_config,
                routing_wall_clock_ms=routing_overhead_ms,
                simulator_excluded_ms=simulator_key_prepare_ms,
                routing_breakdown=routing_breakdown,
                qk_route_info=qk_route_info,
                selected_candidates=selected_candidates,
            )
            qk_route_info["system_cost"] = system_cost

            # 6c. 各策略的 topic 命中率
            #     (包括 lexical 粗选 和 Q-K fine 选)
            selected_topic_ids = qk_route_info.get("selected_topic_ids", [])
            strategy_hits = {}
            for strat_name, sr in strategy_results.items():
                ranked_nodes = sr["ranked_nodes"]
                strategy_hits[strat_name] = {
                    # any_relevant: 选中的 topic 是否包含至少1个 GT turn
                    "top1_any_relevant_hit": sr["top_node"] in relevant_units,
                    "top2_any_relevant_hit": any(
                        node in relevant_units for node in ranked_nodes[:2]
                    ),
                    # dominant: 选中的 topic 是否正好是 GT 最相关的那个
                    "top1_dominant_hit": sr["top_node"] == dominant_unit,
                    "top2_dominant_hit": dominant_unit in ranked_nodes[:2],
                }

            selected_topic_hit = any(node in relevant_units for node in selected_topic_ids)

            selected_chunk_details = []
            for cand in selected_candidates:
                start_t = int(cand["start_t"])
                end_t = int(cand["end_t"])
                turn_idx = int(cand["turn_idx"])
                chunk_text = tokenizer.decode(
                    input_ids[0, start_t:end_t],
                    skip_special_tokens=True,
                ).strip()
                turn_text = ""
                if 0 <= turn_idx < len(transcripts):
                    speaker = transcripts[turn_idx].get("speaker", "").strip() or "Speaker"
                    content = transcripts[turn_idx].get("content", "").strip()
                    turn_text = f"{speaker}: {content}"
                selected_chunk_details.append(
                    {
                        "candidate_id": int(cand.get("candidate_id", -1)),
                        "turn_idx": turn_idx,
                        "local_chunk_idx": int(cand["local_chunk_idx"]),
                        "topic_ids": [int(x) for x in cand.get("topic_ids", [])],
                        "transfer_topic_id": int(cand.get("transfer_topic_id", -1)),
                        "transfer_node_id": int(cand.get("transfer_node_id", -1)),
                        "start_t": start_t,
                        "end_t": end_t,
                        "n_tokens": int(cand["n_tokens"]),
                        "score": float(cand["score"]),
                        "prefilter_score": float(cand.get("prefilter_score", -1e9)),
                        "chunk_text": chunk_text,
                        "turn_text": turn_text,
                    }
                )

            qk_route_info = normalize_qk_route_info(
                qk_route_info,
                selected_chunk_details,
            )
            relevant_candidate_survival = build_relevant_candidate_survival(
                routing_artifacts.get("route_candidates", []),
                relevant_turns,
                qk_route_info,
            )
            qk_route_info["relevant_candidate_survival"] = relevant_candidate_survival

            selected_evidence_entries = build_selected_answer_evidence(
                selected_chunk_details,
                qk_route_info,
                args.selected_answer_context_mode,
            )

            # 6d. 答案生成评测 (可选，--eval_answers)
            #     比较 full context 和 selective context 的答案 F1
            answer_eval = None
            gold_answer = query_info.get("answer", "").strip()
            if args.eval_answers and gold_answer:
                answer_eval = build_answer_eval(
                    model,
                    tokenizer,
                    transcripts,
                    query_text,
                    gold_answer,
                    selected_turn_metrics["selected_turns"],
                    max_new_tokens=args.answer_max_new_tokens,
                    selected_answer_turns=qk_route_info.get("ordered_answer_turns"),
                    selected_evidence_entries=selected_evidence_entries,
                    selected_context_mode=args.selected_answer_context_mode,
                    answer_prompt_style=args.answer_prompt_style,
                    answer_evidence_max_entries=args.answer_evidence_max_entries,
                    answer_evidence_max_chars=args.answer_evidence_max_chars,
                    oracle_turns=sorted(relevant_turns),
                    evaluate_oracle_answer=args.eval_oracle_answers,
                    progress_label=f"doc_id={doc_id} query_idx={query_idx}",
                )

            cachegen_full_estimate = build_cachegen_case_metrics(
                cachegen_doc_estimate,
                system_cost,
                answer_eval=answer_eval,
            )
            cachegen_roundtrip_answer = None
            if args.eval_cachegen_roundtrip_answer:
                if not gold_answer:
                    cachegen_roundtrip_answer = {
                        "enabled": True,
                        "status": "skipped",
                        "baseline_type": "cachegen_full_roundtrip_answer",
                        "skip_reason": "missing_gold_answer",
                    }
                else:
                    print(
                        "  CacheGen roundtrip answer eval "
                        f"doc_id={doc_id} query_idx={query_idx}..."
                    )
                    cachegen_roundtrip_answer = build_cachegen_roundtrip_answer_eval(
                        model=model,
                        tokenizer=tokenizer,
                        transcripts=transcripts,
                        query_text=query_text,
                        gold_answer=gold_answer,
                        max_new_tokens=args.answer_max_new_tokens,
                        cost_config=system_cost_config,
                        answer_prompt_style=args.answer_prompt_style,
                        cachegen_model_name=args.cachegen_model_name,
                        quant_level=args.cachegen_quant_level,
                        chunk_size=args.cachegen_chunk_size,
                        include_encode_time=bool(args.cachegen_include_encode_time),
                        segment_count_mode=args.cachegen_segment_count_mode,
                        answer_eval=answer_eval,
                        system_cost=system_cost,
                    )

            # ---- 整理 route_unit_info（供 trace 用） ----
            topic_to_virtual_node_id = {}
            for layout_item in qk_route_info.get("virtual_node_layout", []):
                node_id = int(layout_item.get("node_id", -1))
                for topic_id in layout_item.get("topic_ids", []):
                    topic_to_virtual_node_id[int(topic_id)] = node_id
            relevant_virtual_node_counts = defaultdict(int)
            for topic_id, count in relevant_topic_counts.items():
                virtual_node_id = int(topic_to_virtual_node_id.get(int(topic_id), -1))
                if virtual_node_id >= 0:
                    relevant_virtual_node_counts[virtual_node_id] += int(count)
            relevant_virtual_nodes = sorted(relevant_virtual_node_counts.keys())

            route_unit_info = {
                "topic_nodes": [
                    {
                        "topic_id": int(topic["topic_id"]),
                        "virtual_node_id": int(
                            topic_to_virtual_node_id.get(int(topic["topic_id"]), -1)
                        ),
                        "label": topic["label"],
                        "num_turns": len(topic["turns"]),
                        "is_gap": bool(topic["is_gap"]),
                    }
                    for topic in topic_nodes
                ],
                "virtual_node_layout": qk_route_info.get("virtual_node_layout", []),
                "relevant_unit_counts": {
                    str(k): int(v) for k, v in sorted(relevant_topic_counts.items())
                },
                "relevant_virtual_node_counts": {
                    str(k): int(v) for k, v in sorted(relevant_virtual_node_counts.items())
                },
            }

            # ======================================================
            # Phase 7: 组装结果条目
            # 包含：基本信息 + 路由信息 + 评测指标 + 传输成本
            # ======================================================
            result_entry = {
                # ---- 基本信息 ----
                "doc_id": doc_id,
                "query_idx": query_idx,
                "meeting_id": meeting.get("meeting_id"),
                "query": query_text,
                "answer_preview": query_info.get("answer", "")[:200],
                "num_turns": len(transcripts),
                "total_tokens": int(total_tokens),
                "num_nodes": args.num_nodes,
                "num_route_units": int(len(topic_nodes)),
                "num_virtual_nodes": int(len(qk_route_info.get("virtual_node_layout", []))),
                "node_assignment_mode": args.node_assignment_mode,
                "offline_route_artifact_prep_ms": float(offline_route_artifact_prep_ms),
                "timing_is_first_query": bool(timing_is_first_query),
                # ---- 路由信息 ----
                "relevant_turns": sorted(relevant_turns),
                "routing_unit_type": "topics",
                "relevant_nodes": relevant_units,
                "dominant_relevant_node": int(dominant_unit),
                "relevant_node_counts": route_unit_info["relevant_unit_counts"],
                "relevant_virtual_nodes": relevant_virtual_nodes,
                "relevant_virtual_node_counts": route_unit_info[
                    "relevant_virtual_node_counts"
                ],
                "qk_score_variance": score_variance,
                "qk_score_range": score_range,
                # ---- 策略对比 ----
                "strategy_results": strategy_results,
                "qk_route_info": qk_route_info,
                "route_unit_info": route_unit_info,
                "transfer_accounting": transfer_accounting,
                "virtual_node_transfer_accounting": virtual_node_transfer_accounting,
                "system_cost": system_cost,
                "cachegen_full_estimate": cachegen_full_estimate,
                "cachegen_roundtrip_answer": cachegen_roundtrip_answer,
                "routing_timing": {
                    "offline_route_artifact_prep_ms": float(offline_route_artifact_prep_ms),
                    "online_route_decision_ms": float(routing_overhead_ms),
                    "system_online_route_decision_ms": float(system_routing_overhead_ms),
                    "simulator_key_prepare_excluded_ms": float(simulator_key_prepare_ms),
                    "timing_is_first_query": bool(timing_is_first_query),
                    "query_tokenize_ms": float(query_tokenize_ms),
                    "coarse_topic_routing_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "coarse_topic_routing_ms",
                            0.0,
                        )
                    ),
                    "topic_filter_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "topic_filter_ms",
                            0.0,
                        )
                    ),
                    "candidate_prefilter_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "candidate_prefilter_ms",
                            0.0,
                        )
                    ),
                    "dynamic_candidate_pool_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "dynamic_candidate_pool_ms",
                            0.0,
                        )
                    ),
                    "coarse_segment_gate_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "coarse_segment_gate_ms",
                            0.0,
                        )
                    ),
                    "node_summary_gate_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "node_summary_gate_ms",
                            0.0,
                        )
                    ),
                    "node_summary_prepare_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "node_summary_prepare_ms",
                            0.0,
                        )
                    ),
                    "node_summary_score_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "node_summary_score_ms",
                            0.0,
                        )
                    ),
                    "node_summary_aggregate_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "node_summary_aggregate_ms",
                            0.0,
                        )
                    ),
                    "candidate_key_prepare_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "candidate_key_prepare_ms",
                            0.0,
                        )
                    ),
                    "query_q_prepare_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "query_q_prepare_ms",
                            0.0,
                        )
                    ),
                    "qk_model_inference_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "qk_model_inference_ms",
                            0.0,
                        )
                    ),
                    "qk_score_aggregation_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "qk_score_aggregation_ms",
                            0.0,
                        )
                    ),
                    "qk_scoring_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "qk_scoring_ms",
                            0.0,
                        )
                    ),
                    "qk_total_stage_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "qk_total_stage_ms",
                            0.0,
                        )
                    ),
                    "selection_postprocess_ms": float(
                        qk_route_info.get("routing_timing_breakdown_ms", {}).get(
                            "selection_postprocess_ms",
                            0.0,
                        )
                    ),
                    "route_unaccounted_ms": float(route_unaccounted_ms),
                },
                "answer_eval": answer_eval,
                "selected_chunk_details": selected_chunk_details,
                "selected_evidence_entries": selected_evidence_entries,
                # ---- 路由评测 ----
                "routing_eval": {
                    "selected_unit_strategy": "lexical",
                    "route_selected_unit_hit": selected_topic_hit,
                    "route_selected_turn_hit": selected_turn_metrics["hit"],
                    "route_selected_turn_recall": selected_turn_metrics["recall"],
                    "route_selected_turn_precision": selected_turn_metrics["precision"],
                    "route_selected_turn_f1": selected_turn_metrics["f1"],
                    "route_relevant_candidate_failure_stage": (
                        relevant_candidate_survival.get("failure_stage", "")
                    ),
                    "route_relevant_candidate_first_drop_stage": (
                        relevant_candidate_survival.get("first_drop_stage", "")
                    ),
                    "route_relevant_candidate_first_zero_stage": (
                        relevant_candidate_survival.get("first_zero_stage", "")
                    ),
                    "route_survival_topic_filter_turn_recall": (
                        relevant_candidate_survival.get("topic_filter_turn_recall", 0.0)
                    ),
                    "route_survival_candidate_prefilter_turn_recall": (
                        relevant_candidate_survival.get(
                            "candidate_prefilter_turn_recall", 0.0
                        )
                    ),
                    "route_survival_dynamic_pool_turn_recall": (
                        relevant_candidate_survival.get(
                            "dynamic_candidate_pool_turn_recall", 0.0
                        )
                    ),
                    "route_survival_coarse_gate_turn_recall": (
                        relevant_candidate_survival.get(
                            "coarse_segment_gate_turn_recall", 0.0
                        )
                    ),
                    "route_survival_node_summary_gate_turn_recall": (
                        relevant_candidate_survival.get(
                            "node_summary_gate_turn_recall", 0.0
                        )
                    ),
                    "route_survival_qk_scored_turn_recall": (
                        relevant_candidate_survival.get("qk_scored_turn_recall", 0.0)
                    ),
                    "route_survival_qk_selected_turn_recall": (
                        relevant_candidate_survival.get("qk_selected_turn_recall", 0.0)
                    ),
                    "route_selected_chunk_count": transfer_accounting["selected_chunk_count"],
                    "route_transfer_unit_count": transfer_accounting["unique_transfer_unit_count"],
                    "route_transfer_segment_count": transfer_accounting["transfer_segment_count"],
                    "route_transfer_virtual_node_count": virtual_node_transfer_accounting[
                        "unique_transfer_unit_count"
                    ],
                    "route_transfer_virtual_node_segment_count": virtual_node_transfer_accounting[
                        "transfer_segment_count"
                    ],
                    "route_offline_artifact_prep_ms": float(offline_route_artifact_prep_ms),
                    "route_routing_overhead_ms": float(system_routing_overhead_ms),
                    "route_routing_wall_clock_ms": float(routing_overhead_ms),
                    "route_simulator_key_prepare_excluded_ms": float(
                        simulator_key_prepare_ms
                    ),
                    "route_candidate_prefilter_mode": qk_route_info.get("candidate_prefilter_mode", ""),
                    "route_candidate_prefilter_pool_size": int(
                        qk_route_info.get("candidate_prefilter_pool_size", 0) or 0
                    ),
                    "route_candidate_prefilter_requested_pool_size": int(
                        qk_route_info.get("candidate_prefilter_requested_pool_size", 0) or 0
                    ),
                    "route_candidate_prefilter_prune_ratio": float(
                        qk_route_info.get("candidate_prefilter_prune_ratio", 0.0) or 0.0
                    ),
                    "route_candidate_prefilter_keep_ratio": float(
                        qk_route_info.get("candidate_prefilter_keep_ratio", 0.0) or 0.0
                    ),
                    "route_candidate_prefilter_min_prune_ratio": float(
                        qk_route_info.get("candidate_prefilter_min_prune_ratio", 0.0) or 0.0
                    ),
                    "route_candidate_prefilter_skip_reason": qk_route_info.get(
                        "candidate_prefilter_skip_reason", ""
                    ),
                    "route_dynamic_candidate_pool_budget": bool(
                        qk_route_info.get("dynamic_candidate_pool_budget", False)
                    ),
                    "route_dynamic_candidate_pool_target": int(
                        qk_route_info.get("dynamic_candidate_pool_target", 0) or 0
                    ),
                    "route_dynamic_candidate_pool_prune_ratio": float(
                        qk_route_info.get("dynamic_candidate_pool_prune_ratio", 0.0) or 0.0
                    ),
                    "route_num_candidates_before_dynamic_pool": int(
                        qk_route_info.get("num_candidates_before_dynamic_pool", 0) or 0
                    ),
                    "route_num_candidates_after_dynamic_pool": int(
                        qk_route_info.get("num_candidates_after_dynamic_pool", 0) or 0
                    ),
                    "route_num_candidates_scored_qk": int(
                        qk_route_info.get("num_candidates_scored_qk", 0) or 0
                    ),
                    "route_coarse_segment_gate": qk_route_info.get(
                        "route_coarse_segment_gate", "none"
                    ),
                    "route_coarse_segment_gate_before": int(
                        qk_route_info.get("coarse_segment_gate_before", 0) or 0
                    ),
                    "route_coarse_segment_gate_after": int(
                        qk_route_info.get("coarse_segment_gate_after", 0) or 0
                    ),
                    "route_coarse_segment_gate_prune_ratio": float(
                        qk_route_info.get("coarse_segment_gate_prune_ratio", 0.0) or 0.0
                    ),
                    "route_node_summary_gate": qk_route_info.get(
                        "node_summary_gate", "none"
                    ),
                    "route_node_summary_gate_summary_mode": qk_route_info.get(
                        "node_summary_gate_summary_mode", ""
                    ),
                    "route_node_summary_gate_before": int(
                        qk_route_info.get("node_summary_gate_before", 0) or 0
                    ),
                    "route_node_summary_gate_after": int(
                        qk_route_info.get("node_summary_gate_after", 0) or 0
                    ),
                    "route_node_summary_gate_target_keep": int(
                        qk_route_info.get("node_summary_gate_target_keep", 0) or 0
                    ),
                    "route_node_summary_gate_prune_ratio": float(
                        qk_route_info.get("node_summary_gate_prune_ratio", 0.0) or 0.0
                    ),
                    "route_num_candidates_before_prefilter": int(
                        qk_route_info.get("num_candidates_before_prefilter", 0) or 0
                    ),
                    "route_num_candidates_after_prefilter": int(
                        qk_route_info.get("num_candidates_after_prefilter", 0) or 0
                    ),
                    "route_cache_candidate_keys": bool(
                        qk_route_info.get("cache_candidate_keys", False)
                    ),
                    "route_candidate_key_cache_hits": int(
                        (qk_route_info.get("candidate_key_cache_stats", {}) or {}).get(
                            "candidate_key_cache_hits",
                            0,
                        )
                    ),
                    "route_candidate_key_cache_misses": int(
                        (qk_route_info.get("candidate_key_cache_stats", {}) or {}).get(
                            "candidate_key_cache_misses",
                            0,
                        )
                    ),
                    "route_candidate_key_cache_size": int(
                        (qk_route_info.get("candidate_key_cache_stats", {}) or {}).get(
                            "candidate_key_cache_size",
                            0,
                        )
                    ),
                    # 兼容旧字段名
                    "qk_selected_node_hit": selected_topic_hit,
                    "qk_selected_turn_hit": selected_turn_metrics["hit"],
                    "qk_selected_turn_recall": selected_turn_metrics["recall"],
                    "qk_selected_turn_precision": selected_turn_metrics["precision"],
                    "qk_selected_turn_f1": selected_turn_metrics["f1"],
                    "qk_selected_turns": selected_turn_metrics["selected_turns"],
                    "qk_matched_turns": selected_turn_metrics["matched_turns"],
                    "strategy_hits": strategy_hits,
                },
            }
            results.append(result_entry)

            # ======================================================
            # Trace 导出：单 case 详细 trace（调试 / 汇报用）
            # 当 --trace_doc_id 和 --trace_query_idx 匹配当前 case 时触发
            # 输出到 --trace_output_dir，格式可选 md / json / both
            # ======================================================
            if should_trace_case(args, doc_id, query_idx):
                trace_payload = {
                    "config": {
                        "mainline_profile": getattr(
                            args, "applied_mainline_profile", "manual"
                        ),
                        "node_assignment_mode": args.node_assignment_mode,
                        "topic_node_layout_path": args.topic_node_layout_path,
                        "routing_granularity": "hierarchical",
                        "hier_top_topics": args.hier_top_topics,
                        "hier_top_strategy": "lexical",
                        "hier_topic_score_mode": args.hier_topic_score_mode,
                        "adaptive_topic_rescue": args.adaptive_topic_rescue,
                        "adaptive_topic_rescue_max_topics": args.adaptive_topic_rescue_max_topics,
                        "adaptive_topic_rescue_margin_ratio": args.adaptive_topic_rescue_margin_ratio,
                        "adaptive_topic_rescue_min_top1_ratio": args.adaptive_topic_rescue_min_top1_ratio,
                        "adaptive_topic_rescue_min_score": args.adaptive_topic_rescue_min_score,
                        "candidate_topic_scope": args.candidate_topic_scope,
                        "topic_prototype_turns": args.topic_prototype_turns,
                        "topic_representation_template": args.topic_representation_template,
                        "lexical_label_repeat": args.lexical_label_repeat,
                        "route_chunk_size": args.route_chunk_size,
                        "route_top_k": args.route_top_k,
                        "effective_route_top_k": qk_route_info.get("effective_route_top_k"),
                        "route_candidate_prefilter": args.route_candidate_prefilter,
                        "route_candidate_prefilter_factor": args.route_candidate_prefilter_factor,
                        "route_candidate_prefilter_min_keep": args.route_candidate_prefilter_min_keep,
                        "route_candidate_prefilter_max_keep": args.route_candidate_prefilter_max_keep,
                        "route_candidate_prefilter_keep_ratio": args.route_candidate_prefilter_keep_ratio,
                        "route_candidate_prefilter_min_prune_ratio": args.route_candidate_prefilter_min_prune_ratio,
                        "route_adaptive_prefilter": args.route_adaptive_prefilter,
                        "route_adaptive_coarse_segment_gate": args.route_adaptive_coarse_segment_gate,
                        "route_adaptive_min_keep_ratio": args.route_adaptive_min_keep_ratio,
                        "route_adaptive_max_keep_ratio": args.route_adaptive_max_keep_ratio,
                        "route_adaptive_entropy_temperature": args.route_adaptive_entropy_temperature,
                        "route_adaptive_min_signal_count": args.route_adaptive_min_signal_count,
                        "route_coarse_segment_gate": args.route_coarse_segment_gate,
                        "route_coarse_segment_size": args.route_coarse_segment_size,
                        "route_coarse_segment_keep_ratio": args.route_coarse_segment_keep_ratio,
                        "route_coarse_segment_min_keep": args.route_coarse_segment_min_keep,
                        "route_coarse_segment_max_keep": args.route_coarse_segment_max_keep,
                        "node_summary_gate": args.node_summary_gate,
                        "node_summary_gate_summary_mode": args.node_summary_gate_summary_mode,
                        "node_summary_gate_factor": args.node_summary_gate_factor,
                        "node_summary_gate_min_keep": args.node_summary_gate_min_keep,
                        "node_summary_gate_max_keep": args.node_summary_gate_max_keep,
                        "node_summary_gate_keep_ratio": args.node_summary_gate_keep_ratio,
                        "node_summary_gate_budget_mode": args.node_summary_gate_budget_mode,
                        "node_summary_gate_adaptive_safety_factor": args.node_summary_gate_adaptive_safety_factor,
                        "node_summary_gate_batch_size": args.node_summary_gate_batch_size,
                        "node_summary_gate_populate_key_cache": bool(
                            args.node_summary_gate_populate_key_cache
                        ),
                        "qk_score_batch_size": args.qk_score_batch_size,
                        "qk_aggregation": args.qk_aggregation,
                        "qk_topk": args.qk_topk,
                        "qk_token_pooling": args.qk_token_pooling,
                        "qk_query_topk_ratio": args.qk_query_topk_ratio,
                        "cache_candidate_keys": bool(args.cache_candidate_keys),
                        "cache_query_q": bool(args.cache_query_q),
                        "candidate_topic_scope": args.candidate_topic_scope,
                        "route_selection_mode": args.route_selection_mode,
                        "route_hybrid_core_ratio": args.route_hybrid_core_ratio,
                        "route_hybrid_core_max_per_turn": args.route_hybrid_core_max_per_turn,
                        "route_pack_anchor_count": args.route_pack_anchor_count,
                        "route_pack_support_radius": args.route_pack_support_radius,
                        "route_pack_max_turns": args.route_pack_max_turns,
                        "route_pack_max_candidates": args.route_pack_max_candidates,
                        "route_pack_support_score_ratio": args.route_pack_support_score_ratio,
                        "route_pack_support_same_turn": bool(
                            args.route_pack_support_same_turn
                        ),
                        "topic_balanced_min_per_topic": args.topic_balanced_min_per_topic,
                        "topic_soft_rescue_max_replacements": args.topic_soft_rescue_max_replacements,
                        "topic_soft_rescue_margin_ratio": args.topic_soft_rescue_margin_ratio,
                        "topic_soft_rescue_min_score_ratio": args.topic_soft_rescue_min_score_ratio,
                        "turn_rerank_qk_weight": args.turn_rerank_qk_weight,
                        "turn_rerank_lexical_weight": args.turn_rerank_lexical_weight,
                        "turn_rerank_head_vote_weight": args.turn_rerank_head_vote_weight,
                        "dynamic_route_budget": bool(args.dynamic_route_budget),
                        "dynamic_candidate_pool_budget": bool(
                            args.dynamic_candidate_pool_budget
                        ),
                        "dynamic_candidate_pool_budget_map": args.dynamic_candidate_pool_budget_map,
                        "dynamic_candidate_pool_min_keep": args.dynamic_candidate_pool_min_keep,
                        "query_budget_type": qk_route_info.get("query_budget_type"),
                        "answer_evidence_order": args.answer_evidence_order,
                        "route_per_head": bool(args.route_per_head),
                        "route_neighbor_expand": int(args.route_neighbor_expand),
                    },
                    "target": {"doc_id": int(doc_id), "query_idx": int(query_idx)},
                    "meeting": {
                        "meeting_id": meeting.get("meeting_id"),
                        "num_turns": len(transcripts),
                        "num_topics_raw": len(meeting.get("topic_list", [])),
                        "num_specific_queries": len(meeting.get("specific_query_list", [])),
                    },
                    "query": {
                        "text": query_text,
                        "answer_preview": preview_text(query_info.get("answer", ""), limit=200),
                        "relevant_turns": sorted(relevant_turns),
                        "relevant_turn_count": len(relevant_turns),
                        "relevant_topic_ids": [int(x) for x in relevant_units],
                    },
                    # 用到的核心函数及其输出（用于理解数据流）
                    "functions_used": [
                        {
                            "function": "load_mainline_sample",
                            "purpose": "Load one meeting from prepared QMSum jsonl.",
                            "key_outputs": {
                                "meeting_id": meeting.get("meeting_id"),
                                "num_transcript_turns": len(transcripts),
                                "num_topics_raw": len(meeting.get("topic_list", [])),
                                "num_specific_queries": len(meeting.get("specific_query_list", [])),
                            },
                        },
                        {
                            "function": "build_qmsum_prompt",
                            "purpose": "Flatten meeting turns and record token boundaries.",
                            "key_outputs": {
                                "prompt_char_len": len(prompt_text),
                                "num_turn_boundaries": len(turn_boundaries),
                                "total_tokens": int(total_tokens),
                            },
                        },
                        {
                            "function": "build_mainline_topic_index_qmsum",
                            "purpose": "Build lexical coarse topic representations from topic labels and representative turns.",
                            "key_outputs": {
                                "num_topic_nodes": len(topic_nodes),
                                "topic_repr_turns": topic_index.get("topic_repr_turns", {}),
                            },
                        },
                        {
                            "function": "score_mainline_topic_chunk",
                            "purpose": "Select topic with lexical scoring, then score only that topic's chunks with Q-K.",
                            "key_outputs": {
                            "selected_topic_ids": qk_route_info.get("selected_topic_ids", []),
                            "num_candidates": qk_route_info.get("num_candidates", 0),
                            "num_candidates_scored_qk": qk_route_info.get("num_candidates_scored_qk", 0),
                            "candidate_key_cache_stats": qk_route_info.get(
                                "candidate_key_cache_stats",
                                {},
                            ),
                            "num_selected_candidates": qk_route_info.get("num_selected_candidates", 0),
                            "selected_token_count": qk_route_info.get("selected_token_count", 0),
                        },
                        },
                    ],
                    "topic_nodes": {
                        "all_topic_nodes": route_unit_info["topic_nodes"],
                        "relevant_topic_counts": route_unit_info["relevant_unit_counts"],
                        "topic_repr_turns": topic_index.get("topic_repr_turns", {}),
                        "topic_repr_entries": {
                            str(topic_id): [
                                {
                                    "entry_type": entry.get("entry_type"),
                                    "text_preview": preview_text(entry.get("text", ""), limit=120),
                                    "turn_idx": entry.get("turn_idx"),
                                }
                                for entry in entry_list
                            ]
                            for topic_id, entry_list in topic_index.get("topic_repr_entries", {}).items()
                        },
                    },
                    "top_level_routing": {
                        "strategy_rankings_top5": {
                            strat_name: [int(x) for x in sr.get("ranked_nodes", [])[:5]]
                            for strat_name, sr in strategy_results.items()
                        },
                        "selected_topic_ids": qk_route_info.get("selected_topic_ids", []),
                        "selected_topics": qk_route_info.get("selected_topics", []),
                    },
                    "chunk_routing": {
                        "num_candidates": qk_route_info.get("num_candidates", 0),
                        "num_candidates_after_topic_filter": qk_route_info.get(
                            "num_candidates_after_topic_filter",
                            0,
                        ),
                        "num_candidates_scored_qk": qk_route_info.get("num_candidates_scored_qk", 0),
                        "candidate_key_cache_stats": qk_route_info.get(
                            "candidate_key_cache_stats",
                            {},
                        ),
                        "num_selected_candidates": qk_route_info.get("num_selected_candidates", 0),
                        "selected_candidates_preview": qk_route_info.get("selected_candidates", []),
                    },
                    "evaluation": {
                        "selected_topic_hit": bool(selected_topic_hit),
                        "selected_turn_hit": bool(selected_turn_metrics["hit"]),
                        "matched_turns": selected_turn_metrics["matched_turns"],
                        "selected_turns": selected_turn_metrics["selected_turns"],
                        "turn_recall": float(selected_turn_metrics["recall"]),
                        "turn_precision": float(selected_turn_metrics["precision"]),
                        "turn_f1": float(selected_turn_metrics["f1"]),
                    },
                    "transfer_accounting": transfer_accounting,
                    "virtual_node_transfer_accounting": virtual_node_transfer_accounting,
                    "system_cost": system_cost,
                }
                if cachegen_full_estimate:
                    trace_payload["cachegen_full_estimate"] = cachegen_full_estimate
                if cachegen_roundtrip_answer:
                    trace_payload["cachegen_roundtrip_answer"] = {
                        "status": cachegen_roundtrip_answer.get("status"),
                        "answer_preview": preview_text(
                            cachegen_roundtrip_answer.get("answer", ""),
                            limit=240,
                        ),
                        "answer_f1": float(
                            cachegen_roundtrip_answer.get("answer_f1", 0.0)
                        ),
                        "answer_f1_delta_vs_full": cachegen_roundtrip_answer.get(
                            "answer_f1_delta_vs_full"
                        ),
                        "compressed_mib": cachegen_roundtrip_answer.get(
                            "compressed_mib"
                        ),
                        "estimated_ttft_ms": cachegen_roundtrip_answer.get(
                            "estimated_ttft_ms"
                        ),
                        "measured_decode_ms": cachegen_roundtrip_answer.get(
                            "measured_decode_ms"
                        ),
                        "bad_output": cachegen_roundtrip_answer.get("bad_output"),
                    }
                if answer_eval:
                    trace_payload["answer_eval"] = {
                        "gold_answer_preview": preview_text(answer_eval.get("gold_answer", ""), limit=240),
                        "full_answer_preview": preview_text(answer_eval.get("full_answer", ""), limit=240),
                        "selected_answer_preview": preview_text(answer_eval.get("selected_answer", ""), limit=240),
                        "oracle_answer_preview": preview_text(answer_eval.get("oracle_answer", ""), limit=240),
                        "oracle_answer_available": bool(
                            answer_eval.get("oracle_answer_available", True)
                        ),
                        "full_answer_f1": float(answer_eval.get("full_answer_f1", 0.0)),
                        "selected_answer_f1": float(answer_eval.get("selected_answer_f1", 0.0)),
                        "oracle_answer_f1": float(answer_eval.get("oracle_answer_f1", 0.0)),
                        "answer_f1_delta": float(answer_eval.get("answer_f1_delta", 0.0)),
                        "oracle_answer_f1_delta_vs_full": float(
                            answer_eval.get("oracle_answer_f1_delta_vs_full", 0.0)
                        ),
                        "selected_answer_f1_delta_vs_oracle": float(
                            answer_eval.get("selected_answer_f1_delta_vs_oracle", 0.0)
                        ),
                        "full_context_tokens": int(answer_eval.get("full_context_tokens", 0)),
                        "selected_context_tokens": int(answer_eval.get("selected_context_tokens", 0)),
                        "oracle_context_tokens": int(answer_eval.get("oracle_context_tokens", 0)),
                        "context_token_saving_ratio": float(
                            answer_eval.get("context_token_saving_ratio", 0.0)
                        ),
                    }

                json_path, md_path = write_case_trace(
                    trace_payload,
                    output_dir=args.trace_output_dir,
                    output_format=args.trace_output_format,
                )
                written_trace_paths.append((json_path, md_path))

            # ---- 前几个样本打印详细结果到控制台 ----
            if len(results) <= 3:
                print("\n" + "-" * 60)
                first_query_note = " first-query-timing" if timing_is_first_query else ""
                print(
                    f"[doc {doc_id} / query {query_idx}] "
                    f"{len(transcripts)} turns, {total_tokens} tokens{first_query_note}"
                )
                print(f"  Query: {query_text[:120]!r}")
                print(f"  Relevant topics: {relevant_units}")
                print(f"  Dominant topic: {format_topic_id(dominant_unit)}")
                print(f"  Relevant turns: {sorted(relevant_turns)[:12]}")
                print(f"  Q-K top topic: {format_topic_id(strategy_results['qk']['top_node'])}")
                print(
                    f"  selected topics: {qk_route_info['selected_topic_ids']}, "
                    f"chunks={qk_route_info['num_selected_candidates']}/"
                    f"{qk_route_info['num_candidates_after_topic_filter']}"
                )
                print(
                    "  route selection: "
                    f"{qk_route_info.get('route_selection_mode', 'chunk_topk')}"
                )
                print(
                    "  candidate prefilter: "
                    f"mode={qk_route_info.get('candidate_prefilter_mode', '')}, "
                    f"before={int(qk_route_info.get('num_candidates_before_prefilter', 0) or 0)}, "
                    f"requested_pool={int(qk_route_info.get('candidate_prefilter_requested_pool_size', 0) or 0)}, "
                    f"after={int(qk_route_info.get('num_candidates_after_prefilter', 0) or 0)}, "
                    f"prune={100.0 * float(qk_route_info.get('candidate_prefilter_prune_ratio', 0.0) or 0.0):.1f}%"
                )
                if qk_route_info.get("dynamic_candidate_pool_budget", False):
                    print(
                        "  dynamic candidate pool: "
                        f"before={int(qk_route_info.get('num_candidates_before_dynamic_pool', 0) or 0)}, "
                        f"target={int(qk_route_info.get('dynamic_candidate_pool_target', 0) or 0)}, "
                        f"after={int(qk_route_info.get('num_candidates_after_dynamic_pool', 0) or 0)}, "
                        f"prune={100.0 * float(qk_route_info.get('dynamic_candidate_pool_prune_ratio', 0.0) or 0.0):.1f}%"
                    )
                print(
                    "  coarse segment gate: "
                    f"mode={qk_route_info.get('route_coarse_segment_gate', 'none')}, "
                    f"before={int(qk_route_info.get('coarse_segment_gate_before', 0) or 0)}, "
                    f"after={int(qk_route_info.get('coarse_segment_gate_after', 0) or 0)}, "
                    f"prune={100.0 * float(qk_route_info.get('coarse_segment_gate_prune_ratio', 0.0) or 0.0):.1f}%"
                )
                print(
                    "  node summary gate: "
                    f"mode={qk_route_info.get('node_summary_gate', 'none')}, "
                    f"summary={qk_route_info.get('node_summary_gate_summary_mode', '')}, "
                    f"score_pool={qk_route_info.get('node_summary_gate_score_pooling', '')}, "
                    f"before={int(qk_route_info.get('node_summary_gate_before', 0) or 0)}, "
                    f"after={int(qk_route_info.get('node_summary_gate_after', 0) or 0)}, "
                    f"target={int(qk_route_info.get('node_summary_gate_target_keep', 0) or 0)}, "
                    f"prune={100.0 * float(qk_route_info.get('node_summary_gate_prune_ratio', 0.0) or 0.0):.1f}%, "
                    f"budget={qk_route_info.get('node_summary_gate_budget_mode', '')}:"
                    f"{qk_route_info.get('node_summary_gate_budget_reason', '')}, "
                    f"nonfinite={int(qk_route_info.get('node_summary_gate_non_finite_score_count', 0) or 0)}"
                )
                print(
                    f"  exact Q-K candidates scored: "
                    f"{int(qk_route_info.get('num_candidates_scored_qk', 0) or 0)}"
                )
                cache_stats = qk_route_info.get("candidate_key_cache_stats", {}) or {}
                if qk_route_info.get("cache_candidate_keys", False):
                    print(
                        "  candidate-key cache: "
                        f"hits={int(cache_stats.get('candidate_key_cache_hits', 0))}, "
                        f"misses={int(cache_stats.get('candidate_key_cache_misses', 0))}, "
                        f"size={int(cache_stats.get('candidate_key_cache_size', 0))}"
                    )
                print(
                    f"  matched turns: {selected_turn_metrics['matched_turns'][:8]}, "
                    f"recall={selected_turn_metrics['recall']:.3f}"
                )
                print(
                    "  relevant survival: "
                    f"topic={relevant_candidate_survival.get('topic_filter_turn_recall', 0.0):.3f}, "
                    f"prefilter={relevant_candidate_survival.get('candidate_prefilter_turn_recall', 0.0):.3f}, "
                    f"dynpool={relevant_candidate_survival.get('dynamic_candidate_pool_turn_recall', 0.0):.3f}, "
                    f"gate={relevant_candidate_survival.get('coarse_segment_gate_turn_recall', 0.0):.3f}, "
                    f"nodegate={relevant_candidate_survival.get('node_summary_gate_turn_recall', 0.0):.3f}, "
                    f"qk_selected={relevant_candidate_survival.get('qk_selected_turn_recall', 0.0):.3f}, "
                    f"fail={relevant_candidate_survival.get('failure_stage', '')}"
                )
                print(
                    "  qk oracle: "
                    f"oracle@budget={relevant_candidate_survival.get('qk_oracle_turn_recall_at_selected_budget', 0.0):.3f}, "
                    f"selector_gap={relevant_candidate_survival.get('qk_selector_gap', 0.0):.3f}, "
                    f"first_gold_rank={int(relevant_candidate_survival.get('qk_first_relevant_rank', 0) or 0)}, "
                    f"diag={relevant_candidate_survival.get('qk_oracle_diagnosis', '')}"
                )
                print(
                    "  routing breakdown: "
                    f"query_tok={query_tokenize_ms:.2f} ms, "
                    f"coarse={float(routing_breakdown.get('coarse_topic_routing_ms', 0.0)):.2f} ms, "
                    f"dynpool={float(routing_breakdown.get('dynamic_candidate_pool_ms', 0.0)):.2f} ms, "
                    f"gate={float(routing_breakdown.get('coarse_segment_gate_ms', 0.0)):.2f} ms, "
                    f"nodegate={float(routing_breakdown.get('node_summary_gate_ms', 0.0)):.2f} ms, "
                    f"sketchprep={float(routing_breakdown.get('node_summary_prepare_ms', 0.0)):.2f} ms, "
                    f"prep={float(routing_breakdown.get('candidate_key_prepare_ms', 0.0)):.2f} ms, "
                    f"qk={float(routing_breakdown.get('qk_scoring_ms', 0.0)):.2f} ms, "
                    f"other={route_unaccounted_ms:.2f} ms"
                )
                print(
                    f"  transfer units={transfer_accounting['unique_transfer_unit_count']}, "
                    f"segments={transfer_accounting['transfer_segment_count']}, "
                    f"selected_tokens={transfer_accounting['selected_token_count']}"
                )
                print(
                    f"  virtual nodes={virtual_node_transfer_accounting['unique_transfer_unit_count']}, "
                    f"node_segments={virtual_node_transfer_accounting['transfer_segment_count']}"
                )
                print(
                    f"  est fetch={system_cost['selected']['fetch_latency_ms']:.2f} ms, "
                    f"est ttft={system_cost['selected']['estimated_ttft_ms']:.2f} ms, "
                    f"system routing={system_cost['routing_overhead_ms']:.2f} ms, "
                    f"routing wall={system_cost['routing_wall_clock_ms']:.2f} ms, "
                    f"sim excluded={system_cost['routing_simulator_excluded_ms']:.2f} ms, "
                    f"offline prep={offline_route_artifact_prep_ms:.2f} ms"
                )
                if answer_eval:
                    print(
                        f"  answer F1 full={answer_eval['full_answer_f1']:.3f}, "
                        f"selected={answer_eval['selected_answer_f1']:.3f}, "
                        f"oracle={answer_eval.get('oracle_answer_f1', 0.0):.3f}, "
                        f"ctx saving={100 * answer_eval['context_token_saving_ratio']:.1f}%"
                    )
                if cachegen_full_estimate and cachegen_full_estimate.get("status") == "ok":
                    print(
                        "  cachegen-full est: "
                        f"compressed={cachegen_full_estimate['compressed_mib']:.1f} MiB, "
                        f"ttft={cachegen_full_estimate['estimated_ttft_ms']:.2f} ms, "
                        f"selected_delta={cachegen_full_estimate['selected_vs_cachegen_ttft_delta_ms']:.2f} ms"
                    )
                if cachegen_roundtrip_answer:
                    if cachegen_roundtrip_answer.get("status") == "ok":
                        print(
                            "  cachegen roundtrip answer: "
                            f"F1={cachegen_roundtrip_answer.get('answer_f1', 0.0):.3f}, "
                            f"compressed={cachegen_roundtrip_answer.get('compressed_mib', 0.0):.1f} MiB, "
                            f"decode={cachegen_roundtrip_answer.get('measured_decode_ms', 0.0):.2f} ms, "
                            f"ttft={cachegen_roundtrip_answer.get('estimated_ttft_ms', 0.0):.2f} ms"
                        )
                    else:
                        print(
                            "  cachegen roundtrip answer: "
                            f"{cachegen_roundtrip_answer.get('status', 'error')} "
                            f"{cachegen_roundtrip_answer.get('error_type', '')}: "
                            f"{cachegen_roundtrip_answer.get('error', cachegen_roundtrip_answer.get('skip_reason', ''))}"
                        )

            # ---- 释放 GPU 内存 ----
            del query_ids
            torch.cuda.empty_cache()

        del input_ids, kv_3d
        torch.cuda.empty_cache()

    # ==================================================================
    # 汇总 & 输出
    # ==================================================================
    if not results:
        print("No results")
        return

    # ---- 构建汇总统计 ----
    summary_payload = build_summary_payload(results, args, args.num_nodes)
    # ---- 打印控制台汇总 ----
    print_summary(summary_payload, args, args.num_nodes)

    # ---- 保存结果文件 ----
    if not args.no_main_output:
        output_suffix = build_output_suffix(args)
        out_path = f"outputs/qmsum_sim_{output_suffix}.json"
        case_tsv_path = f"outputs/qmsum_case_summary_{output_suffix}.tsv"
        answer_log_path = f"outputs/qmsum_answer_log_{output_suffix}.jsonl"
        answer_md_path = f"outputs/qmsum_answer_log_{output_suffix}.md"
        os.makedirs("outputs", exist_ok=True)

        # 完整 JSON（包含所有条目 + 配置 + 汇总）
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "dataset": args.dataset,
                        "entrypoint": "qmsum_mainline.py",
                        "mainline_profile": getattr(
                            args, "applied_mainline_profile", "manual"
                        ),
                        "data_path": args.data_path,
                        "num_nodes": args.num_nodes,
                        "node_assignment_mode": args.node_assignment_mode,
                        "topic_node_layout_path": args.topic_node_layout_path,
                        "routing_granularity": "hierarchical",
                        "hier_top_topics": args.hier_top_topics,
                        "hier_top_strategy": "lexical",
                        "hier_topic_score_mode": args.hier_topic_score_mode,
                        "adaptive_topic_rescue": args.adaptive_topic_rescue,
                        "adaptive_topic_rescue_max_topics": args.adaptive_topic_rescue_max_topics,
                        "adaptive_topic_rescue_margin_ratio": args.adaptive_topic_rescue_margin_ratio,
                        "adaptive_topic_rescue_min_top1_ratio": args.adaptive_topic_rescue_min_top1_ratio,
                        "adaptive_topic_rescue_min_score": args.adaptive_topic_rescue_min_score,
                        "candidate_topic_scope": args.candidate_topic_scope,
                        "topic_prototype_turns": args.topic_prototype_turns,
                        "topic_representation_template": args.topic_representation_template,
                        "lexical_label_repeat": args.lexical_label_repeat,
                        "route_chunk_size": args.route_chunk_size,
                        "route_top_k": args.route_top_k,
                        "route_candidate_prefilter": args.route_candidate_prefilter,
                        "route_candidate_prefilter_factor": args.route_candidate_prefilter_factor,
                        "route_candidate_prefilter_min_keep": args.route_candidate_prefilter_min_keep,
                        "route_candidate_prefilter_max_keep": args.route_candidate_prefilter_max_keep,
                        "route_candidate_prefilter_keep_ratio": args.route_candidate_prefilter_keep_ratio,
                        "route_candidate_prefilter_min_prune_ratio": args.route_candidate_prefilter_min_prune_ratio,
                        "route_adaptive_prefilter": args.route_adaptive_prefilter,
                        "route_adaptive_coarse_segment_gate": args.route_adaptive_coarse_segment_gate,
                        "route_adaptive_min_keep_ratio": args.route_adaptive_min_keep_ratio,
                        "route_adaptive_max_keep_ratio": args.route_adaptive_max_keep_ratio,
                        "route_adaptive_entropy_temperature": args.route_adaptive_entropy_temperature,
                        "route_adaptive_min_signal_count": args.route_adaptive_min_signal_count,
                        "route_coarse_segment_gate": args.route_coarse_segment_gate,
                        "route_coarse_segment_size": args.route_coarse_segment_size,
                        "route_coarse_segment_keep_ratio": args.route_coarse_segment_keep_ratio,
                        "route_coarse_segment_min_keep": args.route_coarse_segment_min_keep,
                        "route_coarse_segment_max_keep": args.route_coarse_segment_max_keep,
                        "node_summary_gate": args.node_summary_gate,
                        "node_summary_gate_summary_mode": args.node_summary_gate_summary_mode,
                        "node_summary_gate_factor": args.node_summary_gate_factor,
                        "node_summary_gate_min_keep": args.node_summary_gate_min_keep,
                        "node_summary_gate_max_keep": args.node_summary_gate_max_keep,
                        "node_summary_gate_keep_ratio": args.node_summary_gate_keep_ratio,
                        "node_summary_gate_budget_mode": args.node_summary_gate_budget_mode,
                        "node_summary_gate_adaptive_safety_factor": args.node_summary_gate_adaptive_safety_factor,
                        "node_summary_gate_batch_size": args.node_summary_gate_batch_size,
                        "node_summary_gate_populate_key_cache": bool(
                            args.node_summary_gate_populate_key_cache
                        ),
                        "qk_score_batch_size": args.qk_score_batch_size,
                        "cache_candidate_keys": args.cache_candidate_keys,
                        "cache_query_q": args.cache_query_q,
                        "dynamic_route_budget": args.dynamic_route_budget,
                        "dynamic_summary_top_k": args.dynamic_summary_top_k,
                        "dynamic_detail_top_k": args.dynamic_detail_top_k,
                        "dynamic_balanced_top_k": args.dynamic_balanced_top_k,
                        "answer_evidence_order": args.answer_evidence_order,
                        "selected_answer_context_mode": args.selected_answer_context_mode,
                        "answer_prompt_style": args.answer_prompt_style,
                        "answer_evidence_max_entries": args.answer_evidence_max_entries,
                        "answer_evidence_max_chars": args.answer_evidence_max_chars,
                        "eval_oracle_answers": args.eval_oracle_answers,
                        "light_output": args.light_output,
                        "verbose_config": args.verbose_config,
                        "case_summary_mode": args.case_summary_mode,
                        "no_answer_jsonl": args.no_answer_jsonl,
                        "no_answer_markdown": args.no_answer_markdown,
                        "ttft_model": args.ttft_model,
                        "fetch_bandwidth_gbps": args.fetch_bandwidth_gbps,
                        "control_bandwidth_gbps": args.control_bandwidth_gbps,
                        "per_node_rtt_ms": args.per_node_rtt_ms,
                        "control_rtt_ms": args.control_rtt_ms,
                        "per_segment_overhead_ms": args.per_segment_overhead_ms,
                        "per_rpc_overhead_ms": args.per_rpc_overhead_ms,
                        "candidate_metadata_bytes": args.candidate_metadata_bytes,
                        "query_request_bytes": args.query_request_bytes,
                        "decode_startup_ms": args.decode_startup_ms,
                        "kv_cache_dtype_bytes": args.kv_cache_dtype_bytes,
                        "kv_bytes_per_token": system_cost_config["kv_bytes_per_token"],
                        "eval_cachegen_full": args.eval_cachegen_full,
                        "eval_cachegen_roundtrip_answer": args.eval_cachegen_roundtrip_answer,
                        "cachegen_baseline_type": (
                            "cachegen_full_roundtrip_answer"
                            if args.eval_cachegen_roundtrip_answer
                            else (
                                "cachegen_full_estimated"
                                if args.eval_cachegen_full
                                else ""
                            )
                        ),
                        "cachegen_model_name": args.cachegen_model_name,
                        "cachegen_quant_level": args.cachegen_quant_level,
                        "cachegen_chunk_size": args.cachegen_chunk_size,
                        "cachegen_include_encode_time": args.cachegen_include_encode_time,
                        "cachegen_decode_ms": args.cachegen_decode_ms,
                        "cachegen_segment_count_mode": args.cachegen_segment_count_mode,
                        "route_per_head": args.route_per_head,
                        "route_neighbor_expand": args.route_neighbor_expand,
                        "candidate_topic_scope": args.candidate_topic_scope,
                        "route_selection_mode": args.route_selection_mode,
                        "route_hybrid_core_ratio": args.route_hybrid_core_ratio,
                        "route_hybrid_core_max_per_turn": args.route_hybrid_core_max_per_turn,
                        "route_pack_anchor_count": args.route_pack_anchor_count,
                        "route_pack_support_radius": args.route_pack_support_radius,
                        "route_pack_max_turns": args.route_pack_max_turns,
                        "route_pack_max_candidates": args.route_pack_max_candidates,
                        "route_pack_support_score_ratio": args.route_pack_support_score_ratio,
                        "topic_balanced_min_per_topic": args.topic_balanced_min_per_topic,
                        "topic_soft_rescue_max_replacements": args.topic_soft_rescue_max_replacements,
                        "topic_soft_rescue_margin_ratio": args.topic_soft_rescue_margin_ratio,
                        "topic_soft_rescue_min_score_ratio": args.topic_soft_rescue_min_score_ratio,
                        "turn_rerank_qk_weight": args.turn_rerank_qk_weight,
                        "turn_rerank_lexical_weight": args.turn_rerank_lexical_weight,
                        "turn_rerank_head_vote_weight": args.turn_rerank_head_vote_weight,
                        "qk_aggregation": args.qk_aggregation,
                        "qk_topk": args.qk_topk,
                        "qk_token_pooling": args.qk_token_pooling,
                        "qk_query_topk_ratio": args.qk_query_topk_ratio,
                        "max_queries_per_doc": args.max_queries_per_doc,
                        "query_tokenizer_warmup": args.query_tokenizer_warmup,
                    },
                    "summary": {
                        "n": summary_payload["n"],
                        "avg_turns": summary_payload["avg_turns"],
                        "routing_unit_type": summary_payload["unit_label"],
                        "avg_relevant_nodes": summary_payload["avg_relevant_nodes"],
                        "all_node_relevant_span_rate": summary_payload["all_node_covered"] / summary_payload["n"]
                        if summary_payload["n"] > 0
                        else 0,
                        "qk_avg_score_variance": summary_payload["avg_variance"],
                        "qk_avg_score_range": summary_payload["avg_range"],
                        "qk_high_variance_ratio": summary_payload["high_var"] / summary_payload["n"]
                        if summary_payload["n"] > 0
                        else 0,
                        "per_strategy": summary_payload["per_strategy"],
                        "qk_selected_node_hit_rate": summary_payload["selected_node_hits"] / summary_payload["n"],
                        "qk_selected_turn_hit_rate": summary_payload["selected_turn_hits"] / summary_payload["n"],
                        "qk_avg_selected_turn_recall": summary_payload["avg_selected_turn_recall"],
                        "qk_avg_selected_turn_precision": summary_payload["avg_selected_turn_precision"],
                        "qk_avg_selected_turn_f1": summary_payload["avg_selected_turn_f1"],
                        "avg_selected_chunk_count": summary_payload["avg_selected_chunks"],
                        "avg_selected_token_count": summary_payload["avg_selected_tokens"],
                        "avg_transfer_unit_count": summary_payload["avg_transfer_units"],
                        "avg_transfer_segment_count": summary_payload["avg_transfer_segments"],
                        "avg_global_contiguous_segment_count": summary_payload["avg_global_segments"],
                        "avg_chunks_per_transfer_segment": summary_payload["avg_chunks_per_transfer_segment"],
                        "avg_transfer_coalescing_gain": summary_payload["avg_coalescing_gain"],
                        "avg_virtual_node_transfer_count": summary_payload.get(
                            "avg_virtual_node_transfer_units", 0.0
                        ),
                        "avg_virtual_node_transfer_segment_count": summary_payload.get(
                            "avg_virtual_node_transfer_segments", 0.0
                        ),
                        "avg_virtual_node_chunks_per_transfer_segment": summary_payload.get(
                            "avg_virtual_node_chunks_per_transfer_segment", 0.0
                        ),
                        "avg_virtual_node_transfer_coalescing_gain": summary_payload.get(
                            "avg_virtual_node_coalescing_gain", 0.0
                        ),
                        "routing_timing_summary": summary_payload.get(
                            "routing_timing_summary"
                        ),
                        "relevant_candidate_survival_summary": summary_payload.get(
                            "relevant_candidate_survival_summary"
                        ),
                        "system_cost_summary": summary_payload.get("system_cost_summary"),
                        "system_cost_steady_state_summary": summary_payload.get(
                            "system_cost_steady_state_summary"
                        ),
                        "cachegen_full_summary": summary_payload.get(
                            "cachegen_full_summary"
                        ),
                        "cachegen_roundtrip_answer_summary": summary_payload.get(
                            "cachegen_roundtrip_answer_summary"
                        ),
                        "answer_summary": summary_payload.get("answer_summary"),
                    },
                    "details": [] if args.light_output else results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved to {out_path}")

        # 逐 case TSV（方便 Excel 查看和对比）
        write_case_summary_tsv(
            results,
            case_tsv_path,
            mode=args.case_summary_mode,
        )
        print(f"Case summary TSV saved to {case_tsv_path} (mode={args.case_summary_mode})")

        # 答案对比日志（如果做了答案评测）
        if args.eval_answers:
            if args.no_answer_jsonl:
                print("Answer log skipped (--no_answer_jsonl)")
            else:
                write_case_answer_log(results, answer_log_path)
                print(f"Answer log saved to {answer_log_path}")
            if args.no_answer_markdown:
                print("Answer markdown skipped (--no_answer_markdown)")
            else:
                write_case_answer_markdown(results, answer_md_path)
                print(f"Answer markdown saved to {answer_md_path}")

    # ---- 打印 trace 文件路径 ----
    for json_path, md_path in written_trace_paths:
        if json_path:
            print(f"Trace JSON saved to {json_path}")
        if md_path:
            print(f"Trace Markdown saved to {md_path}")


def build_arg_parser():
    """构建命令行参数解析器。

    注意：这个 mainline 入口强制 routing_granularity="hierarchical"
    和 hier_top_strategy="lexical"，不接受其他选择。
    如需多策略对比，请使用 qmsum_sim.py。
    """
    parser = argparse.ArgumentParser(description="QMSum mainline simulation")
    add_mainline_profile_argument(parser)

    # ---- 数据和模型 ----
    parser.add_argument("--dataset", type=str, default="qmsum",
                        choices=["qmsum", "hotpotqa"],
                        help="dataset adapter used by the mainline runner")
    parser.add_argument("--data_path", type=str, default=None,
                        help="QMSum structured jsonl 路径")
    parser.add_argument("--model_id", type=str, default="~/models/mistral-7b/",
                        help="HuggingFace 模型名或本地路径")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_gpu_memory", type=int, default=48,
                        help="单 GPU 最大显存 (GB)")
    parser.add_argument("--model_loader", type=str, default="fastchat",
                        choices=["fastchat", "hf"],
                        help="fastchat keeps the original Mistral path; hf uses native Transformers loading")
    parser.add_argument("--hf_quantization", type=str, default="none",
                        choices=["none", "4bit", "8bit"],
                        help="optional Transformers bitsandbytes quantization")
    parser.add_argument("--hf_dtype", type=str, default="bf16",
                        choices=["auto", "bf16", "fp16", "fp32"],
                        help="dtype for native HF loading")
    parser.add_argument("--hf_attn_impl", type=str, default="auto",
                        choices=["auto", "eager", "sdpa"],
                        help="attention implementation for native HF loading")
    parser.add_argument("--hf_device_map", type=str, default="auto",
                        choices=["auto", "balanced", "balanced_low_0", "sequential"],
                        help="device_map for native HF loading")

    # ---- 实验范围 ----
    parser.add_argument("--num_nodes", type=int, default=4,
                        help="虚拟节点数（兼容旧字段，主线中实际按 topic 分）")
    parser.add_argument("--scoring_layers", type=str, default=None,
                        help="Q-K 打分用哪些层，逗号分隔，如 '0,8,16,24,31'")
    parser.add_argument("--start_doc", type=int, default=0)
    parser.add_argument("--end_doc", type=int, default=10)
    parser.add_argument("--max_queries_per_doc", type=int, default=2,
                        help="每个 meeting 最多取几个 query，0=全部")
    parser.add_argument("--node_assignment_mode", type=str, default="contiguous",
                        choices=["contiguous", "round_robin", "manual"],
                        help="节点分配模式（兼容旧字段）")
    parser.add_argument("--topic_node_layout_path", type=str, default="",
                        help="optional JSON topic->virtual-node layout when node_assignment_mode=manual")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="超过此 token 数的文档跳过，0=不限制")

    # ---- Topic 粗路由参数 ----
    parser.add_argument("--routing_granularity", type=str, default="hierarchical",
                        help="路由粒度（主线固定为 hierarchical）")
    parser.add_argument("--hier_top_topics", type=int, default=1,
                        help="粗路由选几个 topic")
    parser.add_argument("--hier_top_strategy", type=str, default="lexical",
                        choices=["lexical"],
                        help="粗路由策略（主线固定为 lexical）")
    parser.add_argument("--hier_topic_score_mode", type=str, default="sum",
                        choices=["sum", "max", "topk_mean"],
                        help="topic 内 chunk 分数聚合方式：sum=加和, max=最大值, topk_mean=top-k 平均")
    parser.add_argument("--hier_topic_topk", type=int, default=4,
                        help="topk_mean 聚合时的 k")
    parser.add_argument("--adaptive_topic_rescue", action="store_true", default=False,
                        help="adaptively add one extra lexical topic when top-topic confidence is low")
    parser.add_argument("--adaptive_topic_rescue_max_topics", type=int, default=2,
                        help="maximum topics selected by adaptive_topic_rescue")
    parser.add_argument("--adaptive_topic_rescue_margin_ratio", type=float, default=0.12,
                        help="rescue when top1-top2 score gap is within this fraction of the topic score span")
    parser.add_argument("--adaptive_topic_rescue_min_top1_ratio", type=float, default=1.15,
                        help="rescue when top1 positive score is less than this multiple of top2 positive score")
    parser.add_argument("--adaptive_topic_rescue_min_score", type=float, default=0.0,
                        help="minimum top2 lexical score needed for adaptive topic rescue")

    # ---- Topic 表示参数 ----
    parser.add_argument("--topic_prototype_turns", type=int, default=3,
                        help="每个 topic 采样几个代表性 turn")
    parser.add_argument("--topic_representation_template", type=str, default="basic",
                        choices=["basic", "enhanced"],
                        help="topic 表示模板：basic=label+turns, enhanced=+speaker+span")
    parser.add_argument("--lexical_label_repeat", type=int, default=3,
                        help="BM25 中 topic label 的重复次数（增大权重）")

    # ---- Chunk 细路由参数 ----
    parser.add_argument("--route_chunk_size", type=int, default=128,
                        help="chunk 大小 (tokens)")
    parser.add_argument("--route_top_k", type=int, default=12,
                        help="细路由选几个 chunk")
    parser.add_argument("--route_per_head", action="store_true", default=False,
                        help="是否逐 head 独立选 chunk（Quest 风格，取并集）")
    parser.add_argument("--route_neighbor_expand", type=int, default=0,
                        help="选中 chunk 的邻居扩展数，0=不扩展")
    parser.add_argument("--candidate_topic_scope", type=str, default="selected_topics",
                        choices=["selected_topics", "all_topics"],
                        help="which topic blocks enter the lightweight candidate-summary stage")
    parser.add_argument("--route_selection_mode", type=str, default="chunk_topk",
                        choices=[
                            "chunk_topk",
                            "turn_rerank",
                            "turn_utility",
                            "turn_rank_fusion",
                            "hybrid",
                            "turn_unique",
                            "turn_unique_guard",
                            "turn_unique_soft",
                            "topic_balanced",
                            "topic_soft_rescue",
                            "evidence_pack",
                            "evidence_pack_v2",
                            "evidence_pack_v3",
                        ],
                        help="final evidence selector after exact Q-K scoring")
    parser.add_argument("--route_hybrid_core_ratio", type=float, default=0.50,
                        help="hybrid selector fraction reserved for raw Q-K core chunks")
    parser.add_argument("--route_hybrid_core_max_per_turn", type=int, default=1,
                        help="maximum raw-QK core chunks kept per turn before coverage/backfill")
    parser.add_argument("--route_pack_anchor_count", type=int, default=3,
                        help="evidence_pack high-QK anchor chunk count")
    parser.add_argument("--route_pack_support_radius", type=int, default=1,
                        help="evidence_pack neighboring chunk support radius around anchors")
    parser.add_argument("--route_pack_max_turns", type=int, default=10,
                        help="evidence_pack maximum distinct turns in the final answer context")
    parser.add_argument("--route_pack_max_candidates", type=int, default=12,
                        help="evidence_pack maximum selected candidates charged to transfer")
    parser.add_argument("--route_pack_support_score_ratio", type=float, default=0.55,
                        help="minimum support Q-K score as a fraction of the best anchor score")
    parser.add_argument("--route_pack_min_support_score", type=float, default=-1e9,
                        help="absolute minimum Q-K score for evidence_pack support chunks")
    parser.add_argument("--route_pack_support_same_turn", action="store_true", default=True,
                        help="allow evidence_pack to add one extra support chunk from an anchor turn")
    parser.add_argument("--no_route_pack_support_same_turn", dest="route_pack_support_same_turn",
                        action="store_false",
                        help="disable same-turn support chunks in evidence_pack")
    parser.add_argument("--topic_balanced_min_per_topic", type=int, default=2,
                        help="minimum final selected chunks reserved for each rescued topic")
    parser.add_argument("--topic_soft_rescue_max_replacements", type=int, default=2,
                        help="maximum selected chunks softly replaced to cover missing rescued topics")
    parser.add_argument("--topic_soft_rescue_margin_ratio", type=float, default=0.15,
                        help="selected-score-span tolerance for topic_soft_rescue replacements")
    parser.add_argument("--topic_soft_rescue_min_score_ratio", type=float, default=0.90,
                        help="minimum replacement score ratio versus dropped chunk in topic_soft_rescue")
    parser.add_argument("--turn_rerank_qk_weight", type=float, default=0.65,
                        help="turn_rerank weight for normalized turn Q-K score")
    parser.add_argument("--turn_rerank_lexical_weight", type=float, default=0.25,
                        help="turn_rerank weight for query-token overlap")
    parser.add_argument("--turn_rerank_head_vote_weight", type=float, default=0.10,
                        help="turn_rerank weight for per-head top-k vote ratio")
    parser.add_argument("--turn_utility_top_m", type=int, default=2,
                        help="number of top chunks aggregated into each turn_utility score")
    parser.add_argument("--turn_utility_prefilter_tiebreak", type=int, default=1,
                        choices=[0, 1],
                        help="whether turn_utility uses lexical prefilter score as a tie-break")
    parser.add_argument("--turn_unique_max_replacements", type=int, default=2,
                        help="maximum duplicate-turn chunks replaced by turn_unique_guard")
    parser.add_argument("--turn_unique_replacement_min_ratio", type=float, default=0.85,
                        help="minimum replacement score ratio versus dropped duplicate chunk")
    parser.add_argument("--turn_unique_soft_margin_ratio", type=float, default=0.15,
                        help="selected-score-span fraction used as the turn_unique_soft replacement tolerance")
    parser.add_argument("--route_candidate_prefilter", type=str, default="none",
                        choices=["none", "lexical"],
                        help="Q-K rerank 前是否先做快速候选召回")
    parser.add_argument("--route_candidate_prefilter_factor", type=int, default=4,
                        help="候选召回池大小 = effective_route_top_k * factor")
    parser.add_argument("--route_candidate_prefilter_min_keep", type=int, default=24,
                        help="候选召回后最少保留的 chunk 数")
    parser.add_argument("--route_candidate_prefilter_max_keep", type=int, default=96,
                        help="候选召回后最多保留的 chunk 数，<=0 表示不设上限")
    parser.add_argument("--route_candidate_prefilter_keep_ratio", type=float, default=0.0,
                        help="minimum fraction of topic-filtered candidates to keep in lexical prefilter")
    parser.add_argument("--route_candidate_prefilter_min_prune_ratio", type=float, default=0.0,
                        help="skip lexical prefilter when the requested pool would prune less than this fraction")
    parser.add_argument("--route_adaptive_prefilter", action="store_true", default=False,
                        help="adapt lexical prefilter keep ratio from cheap-score uncertainty")
    parser.add_argument("--route_adaptive_coarse_segment_gate", action="store_true", default=False,
                        help="adapt coarse segment gate keep ratio from segment-score uncertainty")
    parser.add_argument("--route_adaptive_min_keep_ratio", type=float, default=0.45,
                        help="minimum candidate ratio kept by adaptive survival when cheap scores are confident")
    parser.add_argument("--route_adaptive_max_keep_ratio", type=float, default=0.90,
                        help="maximum candidate ratio kept by adaptive survival when cheap scores are uncertain")
    parser.add_argument("--route_adaptive_entropy_temperature", type=float, default=0.25,
                        help="softmax temperature for entropy-based candidate uncertainty")
    parser.add_argument("--route_adaptive_min_signal_count", type=int, default=4,
                        help="skip adaptive pruning if too few candidates have positive cheap scores")
    parser.add_argument("--route_coarse_segment_gate", type=str, default="none",
                        choices=["none", "lexical"],
                        help="optional coarse segment gate before exact Q-K")
    parser.add_argument("--route_coarse_segment_size", type=int, default=4,
                        help="number of neighboring candidates per coarse gate segment")
    parser.add_argument("--route_coarse_segment_keep_ratio", type=float, default=0.5,
                        help="candidate ratio to keep after coarse segment gate")
    parser.add_argument("--route_coarse_segment_min_keep", type=int, default=48,
                        help="minimum candidates to keep after coarse segment gate")
    parser.add_argument("--route_coarse_segment_max_keep", type=int, default=0,
                        help="maximum candidates to keep after coarse segment gate, <=0 disables cap")
    parser.add_argument("--node_summary_gate", type=str, default="none",
                        choices=["none", "qk_sketch"],
                        help="optional remote-node summary/sketch gate before exact Q-K")
    parser.add_argument("--node_summary_gate_summary_mode", type=str, default="mean_key",
                        choices=["mean_key", "max_key", "max_norm_key", "multi_key", "mean_peak_boundary", "quest_minmax", "minmax_key"],
                        help="how each candidate KV block is summarized for node-summary scoring")
    parser.add_argument("--node_summary_gate_factor", type=int, default=4,
                        help="keep at least route_top_k * factor candidates after node summary gate")
    parser.add_argument("--node_summary_gate_min_keep", type=int, default=24,
                        help="minimum candidates to keep after node summary gate")
    parser.add_argument("--node_summary_gate_max_keep", type=int, default=0,
                        help="maximum candidates to keep after node summary gate, <=0 disables cap")
    parser.add_argument("--node_summary_gate_keep_ratio", type=float, default=0.0,
                        help="minimum fraction of candidates to keep after node summary gate")
    parser.add_argument("--node_summary_gate_budget_mode", type=str, default="fixed",
                        choices=["fixed", "adaptive", "score_only"],
                        help="fixed/adaptive prune candidates; score_only computes sketch scores without pruning")
    parser.add_argument("--node_summary_gate_adaptive_safety_factor", type=float, default=4.0,
                        help="adaptive gate keeps at least route_top_k * this factor before looking for score elbow")
    parser.add_argument("--node_summary_gate_batch_size", type=int, default=64,
                        help="batch size for scoring candidate KV sketches in node summary gate")
    parser.add_argument("--node_summary_gate_populate_key_cache", action="store_true", default=False,
                        help="let node summary sketch construction populate the exact-QK full-key cache")
    parser.add_argument("--qk_score_batch_size", type=int, default=32,
                        help="exact Q-K scoring batch size over candidate chunks")
    parser.add_argument("--cache_candidate_keys", action="store_true", default=False,
                        help="Cache candidate stacked-K tensors within one document and reuse them across queries")
    parser.add_argument("--cache_query_q", dest="cache_query_q", action="store_true", default=False,
                        help="Cache query Q tensors once per query for exact Q-K scoring")
    parser.add_argument("--no_cache_query_q", dest="cache_query_q", action="store_false",
                        help="Disable query-Q caching for exact Q-K scoring")
    parser.add_argument("--dynamic_route_budget", action="store_true", default=False,
                        help="根据 query 类型动态调整 fine-stage chunk budget")
    parser.add_argument("--dynamic_summary_top_k", type=int, default=16,
                        help="summary/discussion 类 query 使用的 chunk top-k")
    parser.add_argument("--dynamic_detail_top_k", type=int, default=8,
                        help="decision/detail 类 query 使用的 chunk top-k")
    parser.add_argument("--dynamic_balanced_top_k", type=int, default=12,
                        help="无法明确分类的 query 使用的 chunk top-k")
    parser.add_argument("--dynamic_candidate_pool_budget", action="store_true", default=False,
                        help="optionally shrink exact-QK candidate pool by query type before coarse gate")
    parser.add_argument("--dynamic_candidate_pool_budget_map", type=str,
                        default="summary:64,detail:48,balanced:48,default:48",
                        help="comma-separated query-type candidate pool caps, e.g. summary:64,detail:48")
    parser.add_argument("--dynamic_candidate_pool_min_keep", type=int, default=24,
                        help="minimum exact-QK candidates to keep when dynamic candidate pool is enabled")
    parser.add_argument("--answer_evidence_order", type=str, default="time",
                        choices=["time", "qk", "qk_then_time", "answer_aware"],
                        help="selected answer prompt 的证据 turn 排序方式")
    parser.add_argument("--coarse_lexical_rerank_topk", type=int, default=3,
                        help="prototype lexical 后进入 full-topic lexical rerank 的 topic 数")
    parser.add_argument("--route_diversity_filter", action="store_true", default=False,
                        help="可选压缩模式：对 selected chunks 做去重/多样性过滤")
    parser.add_argument("--route_diversity_max_similarity", type=float, default=0.8,
                        help="diversity 过滤的 Jaccard 相似度阈值")
    parser.add_argument("--route_diversity_keep_ratio", type=float, default=0.6,
                        help="diversity 过滤后保留 chunk 比例")
    parser.add_argument("--route_diversity_min_keep", type=int, default=24,
                        help="diversity 过滤后最少保留 chunk 数")
    parser.add_argument("--qk_aggregation", type=str, default="mean",
                        choices=["mean", "max", "topk_mean"],
                        help="单个 chunk 内 Q-K 分数聚合方式")
    parser.add_argument("--qk_topk", type=int, default=4,
                        help="Q-K 聚合 topk_mean 时的 k")
    parser.add_argument("--qk_token_pooling", type=str, default="mean",
                        choices=["mean", "query_mean_topk", "query_peak_topk"],
                        help="pool raw query-token x chunk-token Q-K logits before head aggregation")
    parser.add_argument("--qk_query_topk_ratio", type=float, default=0.25,
                        help="fraction of query tokens kept by query_*_topk Q-K pooling")

    # ---- Trace 参数 ----
    parser.add_argument("--trace_doc_id", type=int, default=-1,
                        help="导出 trace 的 doc_id，-1=不导出")
    parser.add_argument("--trace_query_idx", type=int, default=-1,
                        help="导出 trace 的 query_idx")
    parser.add_argument("--trace_output_dir", type=str, default="outputs/qmsum_trace")
    parser.add_argument("--trace_output_format", type=str, default="md",
                        choices=["md", "json", "both"],
                        help="trace 输出格式")

    # ---- 答案评测 ----
    parser.add_argument("--eval_answers", action="store_true", default=False,
                        help="是否比较 full vs selective 答案的 F1")
    parser.add_argument("--no_eval_oracle_answers", dest="eval_oracle_answers",
                        action="store_false", default=True,
                        help="skip oracle answer generation while keeping full/selected F1")
    parser.add_argument("--answer_max_new_tokens", type=int, default=96,
                        help="答案生成最大 token 数")
    parser.add_argument("--selected_answer_context_mode", type=str, default="turns",
                        choices=["turns", "chunk_turns", "chunks"],
                        help="selected answer 使用 turns、chunk 对应 turn，还是原始 chunk evidence")
    parser.add_argument("--answer_prompt_style", type=str, default="basic",
                        choices=["basic", "strict", "grounded"],
                        help="答案生成 prompt 模板")
    parser.add_argument("--answer_evidence_max_entries", type=int, default=80,
                        help="evidence prompt 最多放入多少条证据")
    parser.add_argument("--answer_evidence_max_chars", type=int, default=600,
                        help="每条 evidence 最多保留多少字符")

    # ---- 系统成本模型 ----
    parser.add_argument("--ttft_model", type=str, default="active_node_v2",
                        choices=["active_node_v2", "legacy_serial"],
                        help="TTFT model: active_node_v2 is the second-version summary-assisted active KV-node model")
    parser.add_argument("--fetch_bandwidth_gbps", type=float, default=25.0,
                        help="估算 selective/full fetch 时使用的链路带宽 (Gbps)")
    parser.add_argument("--control_bandwidth_gbps", type=float, default=25.0,
                        help="active-node v2 控制面/query-summary/candidate metadata 带宽 (Gbps)")
    parser.add_argument("--per_node_rtt_ms", type=float, default=1.0,
                        help="每访问一个远端 virtual node 的固定往返开销 (ms)")
    parser.add_argument("--control_rtt_ms", type=float, default=1.0,
                        help="active-node v2 控制面 query dispatch / candidate return RTT (ms/node)")
    parser.add_argument("--per_segment_overhead_ms", type=float, default=0.15,
                        help="每个连续传输 segment 的固定打包/调度开销 (ms)")
    parser.add_argument("--per_rpc_overhead_ms", type=float, default=0.05,
                        help="active-node v2 每个控制面 RPC 的固定调度开销 (ms)")
    parser.add_argument("--candidate_metadata_bytes", type=float, default=64.0,
                        help="active-node v2 每个返回候选 id/score 的控制面元数据字节数")
    parser.add_argument("--query_request_bytes", type=float, default=4096.0,
                        help="active-node v2 向节点发送 query/Q routing request 的估算字节数")
    parser.add_argument("--decode_startup_ms", type=float, default=15.0,
                        help="估算 TTFT 时额外加入的解码启动开销 (ms)")
    parser.add_argument("--kv_cache_dtype_bytes", type=float, default=2.0,
                        help="单个 KV 标量占用的字节数，fp16/bf16 通常取 2")
    parser.add_argument("--query_tokenizer_warmup", type=int, default=1,
                        help="Number of query-tokenizer warmup passes before measured routing")

    # ---- CacheGen full-KV baseline ----
    parser.add_argument("--eval_cachegen_full", action="store_true", default=False,
                        help="measure CacheGen full-KV compression and estimated TTFT")
    parser.add_argument("--eval_cachegen_roundtrip_answer", action="store_true", default=False,
                        help="generate answers from CacheGen compressed+decompressed full prompt KV")
    parser.add_argument("--cachegen_model_name", type=str,
                        default="mistral-community/Mistral-7B-v0.2",
                        help="HF model name used by CacheGen quantization config")
    parser.add_argument("--cachegen_quant_level", type=int, default=2,
                        choices=[1, 2, 3],
                        help="CacheGen quantization level")
    parser.add_argument("--cachegen_chunk_size", type=int, default=256,
                        help="CacheGen encoder chunk size")
    parser.add_argument("--cachegen_include_encode_time", type=int, default=0,
                        choices=[0, 1],
                        help="include CacheGen prepare+encode time in estimated TTFT")
    parser.add_argument("--cachegen_decode_ms", type=float, default=0.0,
                        help="optional measured/assumed CacheGen decode time in ms")
    parser.add_argument("--cachegen_segment_count_mode", type=str, default="one",
                        choices=["one", "cachegen_chunks"],
                        help="network segment model for compressed full-KV transfer")

    # ---- 输出控制 ----
    parser.add_argument("--no_main_output", action="store_true", default=False,
                        help="跳过 JSON/TSV 输出，仅打印控制台")
    parser.add_argument("--light_output", action="store_true", default=False,
                        help="write compact JSON without per-case details; TSV is still written")
    parser.add_argument("--verbose_config", action="store_true", default=False,
                        help="print every resolved config argument at startup")
    parser.add_argument("--case_summary_mode", type=str, default="compact",
                        choices=["compact", "full"],
                        help="compact writes high-signal TSV columns; full preserves all diagnostics")
    parser.add_argument("--no_answer_jsonl", action="store_true", default=False,
                        help="skip per-case answer JSONL when --eval_answers is enabled")
    parser.add_argument("--no_answer_markdown", action="store_true", default=False,
                        help="skip per-case answer markdown when --eval_answers is enabled")
    parser.add_argument("--case_summary_tag", type=str, default="",
                        help="输出文件名后缀标签")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    explicit_arg_dests = collect_explicit_arg_dests(parser, sys.argv[1:])
    args = parser.parse_args()
    args = apply_mainline_profile(args, explicit_arg_dests)

    # ---- 主线强制参数：不允许变更 ----
    # 这个 mainline 只做 hierarchical + lexical
    # 多策略对比去 qmsum_sim.py
    args.routing_granularity = "hierarchical"
    args.hier_top_strategy = "lexical"

    # ---- 默认数据路径 ----
    if args.data_path is None:
        if args.dataset == "hotpotqa":
            args.data_path = os.path.join(
                os.path.dirname(__file__),
                "data",
                "hotpotqa",
                "validation.jsonl",
            )
        else:
            args.data_path = os.path.join(
                os.path.dirname(__file__),
                "data",
                "qmsum_structured",
                "train.jsonl",
            )
    else:
        args.data_path = os.path.expanduser(args.data_path)

    args.model_id = os.path.expanduser(args.model_id)

    if not os.path.exists(args.data_path):
        print(f"Data file does not exist: {args.data_path}")
        sys.exit(1)

    # 确保 src 目录在 path 中（用于 import src.utils）
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    main(args)
