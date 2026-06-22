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
)
from qmsum_data import build_qmsum_prompt, load_qmsum_sample, spans_to_turn_set
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


def main(args):
    # ------------------------------------------------------------------
    # 延迟导入：避免循环依赖，同时让 _chunk_qk_scores_per_head 和
    # split_kv 只在真正跑实验时才加载
    # ------------------------------------------------------------------
    from experiment_chunk_split import _chunk_qk_scores_per_head
    from src.utils import define_model_and_tokenizer, split_kv

    # ==================================================================
    # Phase 0: 打印当前实验配置
    # ==================================================================
    print("=" * 70)
    print("QMSum mainline simulation")
    print(f"  mainline_profile: {getattr(args, 'applied_mainline_profile', 'manual')}")
    print(f"  data: {args.data_path}")
    print("  routing_granularity: hierarchical")
    print("  coarse_strategy: lexical")
    print(f"  node_assignment_mode: {args.node_assignment_mode}")
    print(f"  topic_node_layout_path: {args.topic_node_layout_path}")
    print(f"  hier_top_topics: {args.hier_top_topics}")
    print(f"  hier_topic_score_mode: {args.hier_topic_score_mode}")
    print(f"  topic_prototype_turns: {args.topic_prototype_turns}")
    print(f"  topic_representation_template: {args.topic_representation_template}")
    print(f"  lexical_label_repeat: {args.lexical_label_repeat}")
    print(f"  coarse_lexical_rerank_topk: {args.coarse_lexical_rerank_topk}")
    print(f"  route_chunk_size: {args.route_chunk_size}")
    print(f"  route_top_k: {args.route_top_k}")
    print(f"  route_per_head: {args.route_per_head}")
    print(f"  route_neighbor_expand: {args.route_neighbor_expand}")
    print(f"  route_selection_mode: {args.route_selection_mode}")
    print(f"  turn_rerank_qk_weight: {args.turn_rerank_qk_weight}")
    print(f"  turn_rerank_lexical_weight: {args.turn_rerank_lexical_weight}")
    print(f"  turn_rerank_head_vote_weight: {args.turn_rerank_head_vote_weight}")
    print(f"  route_candidate_prefilter: {args.route_candidate_prefilter}")
    print(f"  route_candidate_prefilter_factor: {args.route_candidate_prefilter_factor}")
    print(f"  route_candidate_prefilter_min_keep: {args.route_candidate_prefilter_min_keep}")
    print(f"  route_candidate_prefilter_max_keep: {args.route_candidate_prefilter_max_keep}")
    print(f"  route_candidate_prefilter_keep_ratio: {args.route_candidate_prefilter_keep_ratio}")
    print(f"  route_candidate_prefilter_min_prune_ratio: {args.route_candidate_prefilter_min_prune_ratio}")
    print(f"  route_coarse_segment_gate: {args.route_coarse_segment_gate}")
    print(f"  route_coarse_segment_size: {args.route_coarse_segment_size}")
    print(f"  route_coarse_segment_keep_ratio: {args.route_coarse_segment_keep_ratio}")
    print(f"  route_coarse_segment_min_keep: {args.route_coarse_segment_min_keep}")
    print(f"  route_coarse_segment_max_keep: {args.route_coarse_segment_max_keep}")
    print(f"  qk_score_batch_size: {args.qk_score_batch_size}")
    print(f"  cache_candidate_keys: {args.cache_candidate_keys}")
    print(f"  cache_query_q: {args.cache_query_q}")
    print(f"  dynamic_route_budget: {args.dynamic_route_budget}")
    print(f"  dynamic_summary_top_k: {args.dynamic_summary_top_k}")
    print(f"  dynamic_detail_top_k: {args.dynamic_detail_top_k}")
    print(f"  dynamic_balanced_top_k: {args.dynamic_balanced_top_k}")
    print(f"  dynamic_candidate_pool_budget: {args.dynamic_candidate_pool_budget}")
    print(f"  dynamic_candidate_pool_budget_map: {args.dynamic_candidate_pool_budget_map}")
    print(f"  dynamic_candidate_pool_min_keep: {args.dynamic_candidate_pool_min_keep}")
    print(f"  answer_evidence_order: {args.answer_evidence_order}")
    print(f"  route_diversity_filter: {args.route_diversity_filter}")
    print(f"  route_diversity_max_similarity: {args.route_diversity_max_similarity}")
    print(f"  route_diversity_keep_ratio: {args.route_diversity_keep_ratio}")
    print(f"  route_diversity_min_keep: {args.route_diversity_min_keep}")
    print(f"  selected_answer_context_mode: {args.selected_answer_context_mode}")
    print(f"  answer_prompt_style: {args.answer_prompt_style}")
    print(f"  answer_evidence_max_entries: {args.answer_evidence_max_entries}")
    print(f"  answer_evidence_max_chars: {args.answer_evidence_max_chars}")
    print(f"  fetch_bandwidth_gbps: {args.fetch_bandwidth_gbps}")
    print(f"  per_node_rtt_ms: {args.per_node_rtt_ms}")
    print(f"  per_segment_overhead_ms: {args.per_segment_overhead_ms}")
    print(f"  decode_startup_ms: {args.decode_startup_ms}")
    print(f"  kv_cache_dtype_bytes: {args.kv_cache_dtype_bytes}")
    print(f"  eval_cachegen_full: {args.eval_cachegen_full}")
    if args.eval_cachegen_full:
        print("  cachegen_baseline_type: cachegen_full_estimated")
        print(f"  cachegen_model_name: {args.cachegen_model_name}")
        print(f"  cachegen_quant_level: {args.cachegen_quant_level}")
        print(f"  cachegen_chunk_size: {args.cachegen_chunk_size}")
        print(f"  cachegen_include_encode_time: {args.cachegen_include_encode_time}")
        print(f"  cachegen_decode_ms: {args.cachegen_decode_ms}")
        print(f"  cachegen_segment_count_mode: {args.cachegen_segment_count_mode}")
    print(f"  query_tokenizer_warmup: {args.query_tokenizer_warmup}")
    print("=" * 70)

    # ==================================================================
    # Phase 1: 加载模型（Mistral-7B）
    # ==================================================================
    print("\nLoading model...")
    model, tokenizer = define_model_and_tokenizer(
        args.model_id,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
    )
    print("Model loaded\n")
    system_cost_config = build_system_cost_config(model.config, args)
    print(
        "System cost model: "
        f"kv_bytes/token={system_cost_config['kv_bytes_per_token']:.0f}, "
        f"bandwidth={system_cost_config['fetch_bandwidth_gbps']:.1f} Gbps\n"
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
                cid
                for cid in turn_candidate_ids
                if cid in set(qk_route_info.get("selected_candidate_ids", []))
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
            "qk_scored_turn_recall": float(
                stage_by_name.get("qk_scored", {}).get("relevant_turn_recall", 0.0)
            ),
            "qk_selected_turn_recall": float(
                stage_by_name.get("qk_selected", {}).get("relevant_turn_recall", 0.0)
            ),
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
            meeting = load_qmsum_sample(args.data_path, doc_id=doc_id)
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
                )

            cachegen_full_estimate = build_cachegen_case_metrics(
                cachegen_doc_estimate,
                system_cost,
                answer_eval=answer_eval,
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
                        "route_coarse_segment_gate": args.route_coarse_segment_gate,
                        "route_coarse_segment_size": args.route_coarse_segment_size,
                        "route_coarse_segment_keep_ratio": args.route_coarse_segment_keep_ratio,
                        "route_coarse_segment_min_keep": args.route_coarse_segment_min_keep,
                        "route_coarse_segment_max_keep": args.route_coarse_segment_max_keep,
                        "qk_score_batch_size": args.qk_score_batch_size,
                        "cache_candidate_keys": bool(args.cache_candidate_keys),
                        "cache_query_q": bool(args.cache_query_q),
                        "route_selection_mode": args.route_selection_mode,
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
                            "function": "load_qmsum_sample",
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
                if answer_eval:
                    trace_payload["answer_eval"] = {
                        "gold_answer_preview": preview_text(answer_eval.get("gold_answer", ""), limit=240),
                        "full_answer_preview": preview_text(answer_eval.get("full_answer", ""), limit=240),
                        "selected_answer_preview": preview_text(answer_eval.get("selected_answer", ""), limit=240),
                        "oracle_answer_preview": preview_text(answer_eval.get("oracle_answer", ""), limit=240),
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
                    f"qk_selected={relevant_candidate_survival.get('qk_selected_turn_recall', 0.0):.3f}, "
                    f"fail={relevant_candidate_survival.get('failure_stage', '')}"
                )
                print(
                    "  routing breakdown: "
                    f"query_tok={query_tokenize_ms:.2f} ms, "
                    f"coarse={float(routing_breakdown.get('coarse_topic_routing_ms', 0.0)):.2f} ms, "
                    f"dynpool={float(routing_breakdown.get('dynamic_candidate_pool_ms', 0.0)):.2f} ms, "
                    f"gate={float(routing_breakdown.get('coarse_segment_gate_ms', 0.0)):.2f} ms, "
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
                        "dataset": "qmsum",
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
                        "route_coarse_segment_gate": args.route_coarse_segment_gate,
                        "route_coarse_segment_size": args.route_coarse_segment_size,
                        "route_coarse_segment_keep_ratio": args.route_coarse_segment_keep_ratio,
                        "route_coarse_segment_min_keep": args.route_coarse_segment_min_keep,
                        "route_coarse_segment_max_keep": args.route_coarse_segment_max_keep,
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
                        "fetch_bandwidth_gbps": args.fetch_bandwidth_gbps,
                        "per_node_rtt_ms": args.per_node_rtt_ms,
                        "per_segment_overhead_ms": args.per_segment_overhead_ms,
                        "decode_startup_ms": args.decode_startup_ms,
                        "kv_cache_dtype_bytes": args.kv_cache_dtype_bytes,
                        "kv_bytes_per_token": system_cost_config["kv_bytes_per_token"],
                        "eval_cachegen_full": args.eval_cachegen_full,
                        "cachegen_baseline_type": "cachegen_full_estimated"
                        if args.eval_cachegen_full
                        else "",
                        "cachegen_model_name": args.cachegen_model_name,
                        "cachegen_quant_level": args.cachegen_quant_level,
                        "cachegen_chunk_size": args.cachegen_chunk_size,
                        "cachegen_include_encode_time": args.cachegen_include_encode_time,
                        "cachegen_decode_ms": args.cachegen_decode_ms,
                        "cachegen_segment_count_mode": args.cachegen_segment_count_mode,
                        "route_per_head": args.route_per_head,
                        "route_neighbor_expand": args.route_neighbor_expand,
                        "route_selection_mode": args.route_selection_mode,
                        "turn_rerank_qk_weight": args.turn_rerank_qk_weight,
                        "turn_rerank_lexical_weight": args.turn_rerank_lexical_weight,
                        "turn_rerank_head_vote_weight": args.turn_rerank_head_vote_weight,
                        "qk_aggregation": args.qk_aggregation,
                        "qk_topk": args.qk_topk,
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
                        "answer_summary": summary_payload.get("answer_summary"),
                    },
                    "details": results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved to {out_path}")

        # 逐 case TSV（方便 Excel 查看和对比）
        write_case_summary_tsv(results, case_tsv_path)
        print(f"Case summary TSV saved to {case_tsv_path}")

        # 答案对比日志（如果做了答案评测）
        if args.eval_answers:
            write_case_answer_log(results, answer_log_path)
            print(f"Answer log saved to {answer_log_path}")
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
    parser.add_argument("--data_path", type=str, default=None,
                        help="QMSum structured jsonl 路径")
    parser.add_argument("--model_id", type=str, default="~/models/mistral-7b/",
                        help="HuggingFace 模型名或本地路径")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_gpu_memory", type=int, default=48,
                        help="单 GPU 最大显存 (GB)")

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
    parser.add_argument("--route_selection_mode", type=str, default="chunk_topk",
                        choices=["chunk_topk", "turn_rerank"],
                        help="final evidence selector after exact Q-K scoring")
    parser.add_argument("--turn_rerank_qk_weight", type=float, default=0.65,
                        help="turn_rerank weight for normalized turn Q-K score")
    parser.add_argument("--turn_rerank_lexical_weight", type=float, default=0.25,
                        help="turn_rerank weight for query-token overlap")
    parser.add_argument("--turn_rerank_head_vote_weight", type=float, default=0.10,
                        help="turn_rerank weight for per-head top-k vote ratio")
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
    parser.add_argument("--fetch_bandwidth_gbps", type=float, default=25.0,
                        help="估算 selective/full fetch 时使用的链路带宽 (Gbps)")
    parser.add_argument("--per_node_rtt_ms", type=float, default=1.0,
                        help="每访问一个远端 virtual node 的固定往返开销 (ms)")
    parser.add_argument("--per_segment_overhead_ms", type=float, default=0.15,
                        help="每个连续传输 segment 的固定打包/调度开销 (ms)")
    parser.add_argument("--decode_startup_ms", type=float, default=15.0,
                        help="估算 TTFT 时额外加入的解码启动开销 (ms)")
    parser.add_argument("--kv_cache_dtype_bytes", type=float, default=2.0,
                        help="单个 KV 标量占用的字节数，fp16/bf16 通常取 2")
    parser.add_argument("--query_tokenizer_warmup", type=int, default=1,
                        help="Number of query-tokenizer warmup passes before measured routing")

    # ---- CacheGen full-KV baseline ----
    parser.add_argument("--eval_cachegen_full", action="store_true", default=False,
                        help="measure CacheGen full-KV compression and estimated TTFT")
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
