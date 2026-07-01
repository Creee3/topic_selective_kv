"""
Central presets for the QMSum selective-KV mainline.

The project has accumulated many ablation switches.  Keep those switches
available, but make the default "current" path explicit and easy to audit.
"""


def _derive_profile(base_name, **updates):
    profile = dict(MAINLINE_PROFILES[base_name])
    profile.update(updates)
    return profile


MAINLINE_PROFILES = {
    "manual": {},
    "current": {
        # Research assumption: semantic topic nodes are manually/dataset
        # provided. The online problem starts from these deployable nodes.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        # Current execution-side default: cheap lexical pruning, a coarse
        # segment gate, then batched exact Q-K on the survivors.
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": True,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 6,
        "route_candidate_prefilter_min_keep": 48,
        "route_candidate_prefilter_max_keep": 128,
        "route_candidate_prefilter_keep_ratio": 0.0,
        "route_candidate_prefilter_min_prune_ratio": 0.0,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.65,
        "route_coarse_segment_min_keep": 64,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "chunk_topk",
        # Stable answer-side defaults for the recent system-side runs.
        "answer_evidence_order": "qk_then_time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "quality_guard": {
        # Conservative candidate-control path for quality recovery.
        #
        # Recent survival diagnostics showed that many gold-turn candidates were
        # dropped before exact Q-K. Keep the stable routing shape, but avoid the
        # brittle query-type pool cap and only let lexical/coarse gates prune
        # when they remove a meaningful amount while still keeping a large pool.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": False,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 12,
        "route_candidate_prefilter_min_keep": 96,
        "route_candidate_prefilter_max_keep": 256,
        "route_candidate_prefilter_keep_ratio": 0.85,
        "route_candidate_prefilter_min_prune_ratio": 0.20,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.85,
        "route_coarse_segment_min_keep": 128,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "chunk_topk",
        "answer_evidence_order": "qk_then_time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "turn_rerank": {
        # Same candidate-control envelope as current, but final evidence is
        # selected at turn level instead of raw chunk top-k. This tests whether
        # Q-K is useful as a weak signal after aggregation/coverage control.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": True,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 6,
        "route_candidate_prefilter_min_keep": 48,
        "route_candidate_prefilter_max_keep": 128,
        "route_candidate_prefilter_keep_ratio": 0.0,
        "route_candidate_prefilter_min_prune_ratio": 0.0,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.65,
        "route_coarse_segment_min_keep": 64,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "turn_rerank",
        "turn_rerank_qk_weight": 0.65,
        "turn_rerank_lexical_weight": 0.25,
        "turn_rerank_head_vote_weight": 0.10,
        "answer_evidence_order": "time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "hybrid_select": {
        # Structured final selector:
        #   1) keep a raw-QK core for high-confidence evidence
        #   2) add turn-level rerank candidates for coverage
        #   3) backfill by raw Q-K if the budget is not full
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": True,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 6,
        "route_candidate_prefilter_min_keep": 48,
        "route_candidate_prefilter_max_keep": 128,
        "route_candidate_prefilter_keep_ratio": 0.0,
        "route_candidate_prefilter_min_prune_ratio": 0.0,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.65,
        "route_coarse_segment_min_keep": 64,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "hybrid",
        "route_hybrid_core_ratio": 0.50,
        "route_hybrid_core_max_per_turn": 1,
        "turn_rerank_qk_weight": 0.65,
        "turn_rerank_lexical_weight": 0.25,
        "turn_rerank_head_vote_weight": 0.10,
        "answer_evidence_order": "qk_then_time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "adaptive_survival": {
        # Candidate-control experiment:
        #   1) keep lexical top-1 topic routing unchanged
        #   2) replace fixed candidate caps with uncertainty-aware survival
        #   3) keep final raw Q-K chunk selection unchanged
        #
        # This isolates whether the pre-QK funnel can become less brittle
        # without relying on gold labels or dataset-specific fixed caps.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": False,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 5,
        "route_candidate_prefilter_min_keep": 48,
        "route_candidate_prefilter_max_keep": 160,
        "route_candidate_prefilter_keep_ratio": 0.0,
        "route_candidate_prefilter_min_prune_ratio": 0.10,
        "route_adaptive_prefilter": True,
        "route_adaptive_coarse_segment_gate": True,
        "route_adaptive_min_keep_ratio": 0.45,
        "route_adaptive_max_keep_ratio": 0.90,
        "route_adaptive_entropy_temperature": 0.25,
        "route_adaptive_min_signal_count": 4,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.65,
        "route_coarse_segment_min_keep": 64,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "chunk_topk",
        "answer_evidence_order": "qk_then_time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "adaptive_hybrid": {
        # Same uncertainty-aware candidate survival as adaptive_survival, then
        # use the hybrid final selector. Compare against adaptive_survival to
        # separate funnel effects from final evidence-selection effects.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": False,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 5,
        "route_candidate_prefilter_min_keep": 48,
        "route_candidate_prefilter_max_keep": 160,
        "route_candidate_prefilter_keep_ratio": 0.0,
        "route_candidate_prefilter_min_prune_ratio": 0.10,
        "route_adaptive_prefilter": True,
        "route_adaptive_coarse_segment_gate": True,
        "route_adaptive_min_keep_ratio": 0.45,
        "route_adaptive_max_keep_ratio": 0.90,
        "route_adaptive_entropy_temperature": 0.25,
        "route_adaptive_min_signal_count": 4,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.65,
        "route_coarse_segment_min_keep": 64,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "hybrid",
        "route_hybrid_core_ratio": 0.50,
        "route_hybrid_core_max_per_turn": 1,
        "turn_rerank_qk_weight": 0.65,
        "turn_rerank_lexical_weight": 0.25,
        "turn_rerank_head_vote_weight": 0.10,
        "answer_evidence_order": "qk_then_time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "evidence_pack": {
        # Answer-facing evidence organization over the current candidate funnel.
        #
        # Unlike adaptive_survival, this does not widen the pre-QK candidate pool.
        # It tests whether selected chunks should be organized as anchor+support
        # evidence before answer generation, while still charging transferred KV
        # for every selected support chunk.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": True,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 6,
        "route_candidate_prefilter_min_keep": 48,
        "route_candidate_prefilter_max_keep": 128,
        "route_candidate_prefilter_keep_ratio": 0.0,
        "route_candidate_prefilter_min_prune_ratio": 0.0,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.65,
        "route_coarse_segment_min_keep": 64,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "evidence_pack",
        "route_pack_anchor_count": 3,
        "route_pack_support_radius": 1,
        "route_pack_max_turns": 10,
        "route_pack_max_candidates": 12,
        "route_pack_support_score_ratio": 0.55,
        "route_pack_support_same_turn": True,
        "answer_evidence_order": "qk_then_time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "evidence_pack_v2": {
        # Support-first evidence organization over the current candidate
        # funnel. This keeps the same pre-QK path as "current", then changes
        # only the final selected-evidence layout: Q-K anchors first, local
        # support around those anchors second, and broad turn coverage last.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": True,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 6,
        "route_candidate_prefilter_min_keep": 48,
        "route_candidate_prefilter_max_keep": 128,
        "route_candidate_prefilter_keep_ratio": 0.0,
        "route_candidate_prefilter_min_prune_ratio": 0.0,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.65,
        "route_coarse_segment_min_keep": 64,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "evidence_pack_v2",
        "route_pack_anchor_count": 3,
        "route_pack_support_radius": 1,
        "route_pack_max_turns": 8,
        "route_pack_max_candidates": 12,
        "route_pack_support_score_ratio": 0.0,
        "route_pack_support_same_turn": True,
        "answer_evidence_order": "qk_then_time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "evidence_pack_v3": {
        # Conservative evidence support. Keep the current top-QK evidence set
        # size, then swap only a few low-value duplicate-turn tail chunks for
        # local anchor support. This tests support without the recall loss seen
        # in evidence_pack_v2.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": True,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 6,
        "route_candidate_prefilter_min_keep": 48,
        "route_candidate_prefilter_max_keep": 128,
        "route_candidate_prefilter_keep_ratio": 0.0,
        "route_candidate_prefilter_min_prune_ratio": 0.0,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.65,
        "route_coarse_segment_min_keep": 64,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "evidence_pack_v3",
        "route_pack_anchor_count": 3,
        "route_pack_support_radius": 1,
        "route_pack_max_turns": 12,
        "route_pack_max_candidates": 12,
        "route_pack_support_score_ratio": 0.0,
        "route_pack_support_same_turn": True,
        "answer_evidence_order": "qk_then_time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "adaptive_evidence_pack": {
        # Diagnostic branch: use adaptive candidate survival, but replace raw
        # chunk_topk/hybrid with answer-facing anchor+support packing.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": True,
        "dynamic_summary_top_k": 16,
        "dynamic_detail_top_k": 12,
        "dynamic_balanced_top_k": 12,
        "dynamic_candidate_pool_budget": False,
        "dynamic_candidate_pool_budget_map": "summary:96,detail:48,balanced:56,default:56",
        "dynamic_candidate_pool_min_keep": 24,
        "route_candidate_prefilter": "lexical",
        "route_candidate_prefilter_factor": 5,
        "route_candidate_prefilter_min_keep": 48,
        "route_candidate_prefilter_max_keep": 160,
        "route_candidate_prefilter_keep_ratio": 0.0,
        "route_candidate_prefilter_min_prune_ratio": 0.10,
        "route_adaptive_prefilter": True,
        "route_adaptive_coarse_segment_gate": True,
        "route_adaptive_min_keep_ratio": 0.45,
        "route_adaptive_max_keep_ratio": 0.90,
        "route_adaptive_entropy_temperature": 0.25,
        "route_adaptive_min_signal_count": 4,
        "route_coarse_segment_gate": "lexical",
        "route_coarse_segment_size": 4,
        "route_coarse_segment_keep_ratio": 0.65,
        "route_coarse_segment_min_keep": 64,
        "route_coarse_segment_max_keep": 0,
        "qk_score_batch_size": 64,
        "cache_candidate_keys": True,
        "cache_query_q": True,
        "route_selection_mode": "evidence_pack",
        "route_pack_anchor_count": 3,
        "route_pack_support_radius": 1,
        "route_pack_max_turns": 10,
        "route_pack_max_candidates": 12,
        "route_pack_support_score_ratio": 0.55,
        "route_pack_support_same_turn": True,
        "answer_evidence_order": "qk_then_time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "grounded",
        "answer_max_new_tokens": 96,
        "answer_evidence_max_entries": 80,
        "answer_evidence_max_chars": 600,
    },
    "simple": {
        # Minimal teaching/debugging path: no middle pruning/gating extras.
        "routing_granularity": "hierarchical",
        "hier_top_strategy": "lexical",
        "hier_top_topics": 1,
        "hier_topic_score_mode": "sum",
        "route_chunk_size": 128,
        "route_top_k": 12,
        "route_neighbor_expand": 0,
        "route_per_head": True,
        "dynamic_route_budget": False,
        "dynamic_candidate_pool_budget": False,
        "route_candidate_prefilter": "none",
        "route_coarse_segment_gate": "none",
        "qk_score_batch_size": 32,
        "cache_candidate_keys": False,
        "cache_query_q": False,
        "route_selection_mode": "chunk_topk",
        "answer_evidence_order": "time",
        "selected_answer_context_mode": "turns",
        "answer_prompt_style": "basic",
        "answer_max_new_tokens": 96,
    },
}


MAINLINE_PROFILES["turn_unique"] = _derive_profile(
    "current",
    # Same pre-QK funnel as current, but final selection avoids spending the
    # answer budget on many chunks from the same transcript turn.
    route_selection_mode="turn_unique",
    route_pack_anchor_count=3,
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["turn_utility"] = _derive_profile(
    "current",
    # Final selector for the advisor-revised framing: lightweight scored
    # chunks are aggregated into a turn/block utility first, then only one full
    # KV chunk per selected turn is fetched. This avoids local threshold tuning
    # and tests whether the bottleneck is raw chunk-level budget competition.
    route_selection_mode="turn_utility",
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["quality_guard_turn_utility"] = _derive_profile(
    "quality_guard",
    # Keep quality_guard's wider candidate-survival envelope, then use the
    # turn/block utility selector that reduced the final Q-K selection gap.
    # This tests the next bottleneck without introducing another threshold
    # search.
    route_selection_mode="turn_utility",
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["global_summary_turn_utility"] = _derive_profile(
    "turn_utility",
    # Advisor-revised architecture probe: do not let lexical top-1 topic be a
    # hard gate. All topic/block candidates expose lightweight lexical summary
    # scores first; the request node keeps a summary-ranked candidate pool,
    # then runs exact Q-K and turn/block utility selection for the final KV
    # fetch. This tests node/block summary routing rather than local top-1
    # topic tuning.
    candidate_topic_scope="all_topics",
    route_candidate_prefilter="lexical",
    route_candidate_prefilter_factor=12,
    route_candidate_prefilter_min_keep=96,
    route_candidate_prefilter_max_keep=256,
    route_candidate_prefilter_keep_ratio=0.0,
    dynamic_candidate_pool_budget=False,
    route_coarse_segment_gate="lexical",
    route_coarse_segment_keep_ratio=0.85,
    route_coarse_segment_min_keep=128,
    route_coarse_segment_max_keep=0,
    route_selection_mode="turn_utility",
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["global_summary_turn_rerank"] = _derive_profile(
    "global_summary_turn_utility",
    # Same all-topic summary candidate pool, but final turn ranking uses the
    # existing turn-level reranker that mixes Q-K, lexical summary overlap, and
    # head-vote consensus. This tests whether the global pool needs a soft
    # metadata prior rather than pure Q-K/turn-utility ordering.
    route_selection_mode="turn_rerank",
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["global_summary_rank_fusion"] = _derive_profile(
    "global_summary_turn_utility",
    # Same advisor-style all-topic candidate pool, but final selection fuses
    # Q-K, lightweight summary, optional node-summary, and head-vote ranks
    # instead of hand-tuned score weights. This is the cleaner architectural
    # probe for "query + remote summary -> selected full KV blocks".
    route_selection_mode="turn_rank_fusion",
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["global_node_summary_rank_fusion"] = _derive_profile(
    "global_summary_rank_fusion",
    # Closest executable probe for the advisor-revised model:
    # all topic/block candidates first expose lightweight KV-key sketches; the
    # request node scores those sketches with query-Q, keeps a bounded candidate
    # set, then exact Q-K plus rank-fusion chooses which full KV blocks to fetch.
    node_summary_gate="qk_sketch",
    node_summary_gate_summary_mode="multi_key",
    node_summary_gate_factor=4,
    node_summary_gate_min_keep=24,
    node_summary_gate_max_keep=0,
    node_summary_gate_keep_ratio=0.0,
    node_summary_gate_budget_mode="adaptive",
    node_summary_gate_adaptive_safety_factor=4.0,
    node_summary_gate_batch_size=64,
    node_summary_gate_populate_key_cache=False,
    cache_query_q=True,
    cache_candidate_keys=True,
)

MAINLINE_PROFILES["global_node_summary_soft_rank_fusion"] = _derive_profile(
    "global_node_summary_rank_fusion",
    # Diagnostic follow-up after hard sketch-gating lost too many gold turns:
    # compute the same remote KV-key sketch scores, but use them only as a soft
    # rank-fusion signal. No full candidate is dropped before exact Q-K.
    node_summary_gate_budget_mode="score_only",
)

MAINLINE_PROFILES["global_quest_minmax_soft_rank_fusion"] = _derive_profile(
    "global_node_summary_soft_rank_fusion",
    # Borrowed from Quest's page-summary implementation: each remote block
    # exposes K_min/K_max metadata, and the request node scores an optimistic
    # query-Q upper bound before exact Q-K. Keep it soft first so the sketch
    # cannot delete relevant blocks before we know whether the signal is useful.
    node_summary_gate_summary_mode="quest_minmax",
    node_summary_gate_budget_mode="score_only",
)

MAINLINE_PROFILES["lexical_top2_turn_utility"] = _derive_profile(
    "turn_utility",
    # Middle ground between hard top-1 topic routing and all-topic routing:
    # fetch candidates from the top-2 lexical topic nodes, then keep the
    # turn/block utility selector. This tests whether the advisor-style
    # multi-node candidate set can fix topic misses without flooding Q-K with
    # global false positives.
    hier_top_topics=2,
    candidate_topic_scope="selected_topics",
    route_selection_mode="turn_utility",
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["lexical_top2_rank_fusion"] = _derive_profile(
    "lexical_top2_turn_utility",
    # Bounded two-topic version of rank-fusion. This checks whether a modest
    # multi-node candidate set plus summary/Q-K rank agreement is better than
    # either hard top-1 routing or opening all topic nodes.
    route_selection_mode="turn_rank_fusion",
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["turn_unique_guard"] = _derive_profile(
    "current",
    # Conservative de-duplication: start from current's Q-K selected chunks and
    # replace only low-score duplicate-turn tail chunks with comparable new
    # turns.  This isolates whether duplicate chunks are waste without throwing
    # away high-QK repeated evidence.
    route_selection_mode="turn_unique_guard",
    route_pack_anchor_count=4,
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["turn_unique_soft"] = _derive_profile(
    "current",
    # Boundary-band de-duplication: keep current's Q-K core, but when the
    # selected tail has duplicate chunks from one answer turn, swap at most one
    # weak duplicate for a new turn if its Q-K score lies inside the local
    # score-uncertainty band. This avoids the negative-score ratio trap in the
    # guard variant.
    route_selection_mode="turn_unique_soft",
    route_pack_anchor_count=4,
    turn_unique_max_replacements=1,
    turn_unique_soft_margin_ratio=0.15,
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["turn_unique_answeraware"] = _derive_profile(
    "turn_unique",
    # Same selected chunks as turn_unique; only the answer-side order changes.
    answer_evidence_order="answer_aware",
)

MAINLINE_PROFILES["turn_unique_guard_answeraware"] = _derive_profile(
    "turn_unique_guard",
    # Same conservative selected chunks as turn_unique_guard; only answer order changes.
    answer_evidence_order="answer_aware",
)

MAINLINE_PROFILES["turn_unique_soft_answeraware"] = _derive_profile(
    "turn_unique_soft",
    # Same soft selected chunks as turn_unique_soft; only answer order changes.
    answer_evidence_order="answer_aware",
)

MAINLINE_PROFILES["adaptive_topic_rescue"] = _derive_profile(
    "current",
    # Keep the stable current fine routing, but widen the top-level topic gate
    # only when lexical top-1 is uncertain. This targets topic_filter misses
    # without always paying the top-2 transfer/Q-K cost.
    adaptive_topic_rescue=True,
    adaptive_topic_rescue_max_topics=2,
    adaptive_topic_rescue_margin_ratio=0.12,
    adaptive_topic_rescue_min_top1_ratio=1.15,
    adaptive_topic_rescue_min_score=0.0,
)

MAINLINE_PROFILES["adaptive_topic_turn_utility"] = _derive_profile(
    "adaptive_topic_rescue",
    # Same adaptive topic widening as soft_rescue, but final evidence is chosen
    # by turn/block utility rather than topic quotas or weak replacements.
    route_selection_mode="turn_utility",
    answer_evidence_order="qk_then_time",
)

MAINLINE_PROFILES["adaptive_topic_balanced"] = _derive_profile(
    "adaptive_topic_rescue",
    # If topic rescue keeps a second possible topic, reserve a tiny final
    # evidence budget for each rescued topic before global Q-K backfill. This
    # tests whether top-2 rescue failed because final global top-k re-collapsed
    # onto the dominant topic.
    route_selection_mode="topic_balanced",
    topic_balanced_min_per_topic=2,
)

MAINLINE_PROFILES["adaptive_topic_soft_rescue"] = _derive_profile(
    "adaptive_topic_rescue",
    # Keep the global Q-K top-k as the default. If a rescued topic is missing
    # from final evidence, replace only weak boundary chunks when the rescued
    # topic candidate is score-close enough. This is a bounded utility
    # correction, not a hard quota.
    route_selection_mode="topic_soft_rescue",
    topic_soft_rescue_max_replacements=2,
    topic_soft_rescue_margin_ratio=0.15,
    topic_soft_rescue_min_score_ratio=0.90,
)

MAINLINE_PROFILES["qk_head_topk"] = _derive_profile(
    "current",
    # Sharpen only the head aggregation after the usual token-level mean.
    # This isolates whether a few strong attention heads are more reliable
    # than averaging every head.
    qk_aggregation="topk_mean",
    qk_topk=4,
    qk_token_pooling="mean",
    qk_query_topk_ratio=0.25,
)

MAINLINE_PROFILES["qk_query_mean_topk"] = _derive_profile(
    "current",
    # Keep each query token's average match to the chunk, then aggregate only
    # the strongest query tokens. This tests whether filler query tokens dilute
    # evidence selection.
    qk_aggregation="mean",
    qk_topk=4,
    qk_token_pooling="query_mean_topk",
    qk_query_topk_ratio=0.25,
)

MAINLINE_PROFILES["qk_query_peak_topk"] = _derive_profile(
    "current",
    # Keep each query token's best local chunk-token match, then aggregate only
    # the strongest query tokens. This is the most sparse "keyword peak" test.
    qk_aggregation="mean",
    qk_topk=4,
    qk_token_pooling="query_peak_topk",
    qk_query_topk_ratio=0.25,
)

MAINLINE_PROFILES["node_summary_gate"] = _derive_profile(
    "current",
    # New compute-symmetric/distributed-node sketch path:
    # each KV node exposes a tiny per-block key sketch first; the request node
    # scores those sketches with its query-Q, then fetches full KV only for the
    # surviving blocks and runs exact Q-K on that smaller pool.
    node_summary_gate="qk_sketch",
    node_summary_gate_summary_mode="mean_key",
    node_summary_gate_factor=4,
    node_summary_gate_min_keep=96,
    node_summary_gate_max_keep=96,
    node_summary_gate_keep_ratio=0.0,
    node_summary_gate_budget_mode="fixed",
    node_summary_gate_adaptive_safety_factor=4.0,
    node_summary_gate_batch_size=64,
    node_summary_gate_populate_key_cache=False,
    cache_query_q=True,
    cache_candidate_keys=True,
)

MAINLINE_PROFILES["node_summary_gate_active"] = _derive_profile(
    "current",
    # Mechanism probe: leave a wider candidate pool for the node-summary sketch
    # stage, then let query-Q-over-sketch prune before exact Q-K. This profile is
    # for ablation, while node_summary_gate is the conservative safe default.
    route_candidate_prefilter="lexical",
    route_candidate_prefilter_factor=12,
    route_candidate_prefilter_min_keep=96,
    route_candidate_prefilter_max_keep=0,
    dynamic_candidate_pool_budget=False,
    route_coarse_segment_gate="none",
    node_summary_gate="qk_sketch",
    node_summary_gate_summary_mode="mean_key",
    node_summary_gate_factor=4,
    node_summary_gate_min_keep=24,
    node_summary_gate_max_keep=0,
    node_summary_gate_keep_ratio=0.0,
    node_summary_gate_budget_mode="adaptive",
    node_summary_gate_adaptive_safety_factor=4.0,
    node_summary_gate_batch_size=64,
    node_summary_gate_populate_key_cache=False,
    cache_query_q=True,
    cache_candidate_keys=True,
)


MAINLINE_PROFILES["node_summary_gate_multi"] = _derive_profile(
    "node_summary_gate_active",
    # Mechanism upgrade over mean_key: expose several lightweight K prototypes
    # per candidate block (mean, high-salience peak, first, last). The scorer
    # uses query-peak pooling so a localized relevant prototype can survive.
    node_summary_gate_summary_mode="multi_key",
)

# Keep the original lexical-top1 line addressable for ablations, but make
# "current" mean the advisor-revised active-node v2 mainline.
MAINLINE_PROFILES["legacy_lexical_v1"] = dict(MAINLINE_PROFILES["current"])
MAINLINE_PROFILES["current"] = _derive_profile(
    "global_quest_minmax_soft_rank_fusion",
    # Mainline v2: all topic/block candidates expose lightweight remote-node
    # summaries first; the request node scores those summaries with query-Q,
    # fuses summary/Q-K ranks, and fetches only selected full KV blocks.
)

FORCED_MAINLINE_VALUES = {
    "routing_granularity": "hierarchical",
    "hier_top_strategy": "lexical",
}


def add_mainline_profile_argument(parser):
    parser.add_argument(
        "--mainline_profile",
        type=str,
        default="manual",
        choices=sorted(MAINLINE_PROFILES.keys()),
        help=(
            "Preset for the QMSum mainline. 'manual' preserves explicit/parser "
            "defaults, 'current' is the advisor-revised active-node v2 path, 'legacy_lexical_v1' preserves the previous lexical-top1 path, "
            "'quality_guard' is a conservative candidate-recall path, and "
            "'turn_rerank' tests turn-level final evidence selection. "
            "'hybrid_select' keeps a raw-QK core plus turn-level coverage. "
            "'turn_unique' keeps current's pre-QK funnel but prefers one "
            "selected chunk per answer turn; 'turn_unique_guard' is the "
            "conservative replacement variant; 'turn_unique_soft' replaces "
            "duplicate-turn tail chunks only inside a local Q-K boundary band; "
            "the '*_answeraware' profiles only change answer-side ordering. "
            "'adaptive_topic_rescue' keeps current's final selector but adds a "
            "second topic only when lexical top-1 confidence is low. "
            "'adaptive_topic_balanced' adds topic-balanced final evidence "
            "selection after topic rescue. "
            "'adaptive_topic_soft_rescue' keeps global Q-K top-k but softly "
            "replaces weak chunks to cover missing rescued topics. "
            "'qk_head_topk', 'qk_query_mean_topk', and 'qk_query_peak_topk' "
            "test sharper Q-K aggregation without changing the candidate funnel. "
            "'global_summary_turn_utility' lets all topic/block candidates enter "
            "a lightweight summary pool before Q-K and turn utility selection. "
            "'global_summary_turn_rerank' adds a soft lexical/head-vote prior "
            "inside that all-topic pool, 'global_summary_rank_fusion' fuses "
            "available signal ranks without fixed weights, and "
            "'lexical_top2_turn_utility'/'lexical_top2_rank_fusion' test "
            "bounded two-topic candidate pools. "
            "'adaptive_survival' uses uncertainty-aware candidate survival, "
            "and 'adaptive_hybrid' combines it with hybrid selection. "
            "'evidence_pack' organizes selected chunks as anchor+support "
            "evidence, 'evidence_pack_v2' reserves explicit local-support "
            "budget around Q-K anchors, 'evidence_pack_v3' makes conservative "
            "support swaps, and 'adaptive_evidence_pack' combines that with "
            "adaptive candidate survival. "
            "'node_summary_gate' adds a query-Q over remote-node KV sketches "
            "stage before exact Q-K, modeling summary-first selective fetch; "
            "'node_summary_gate_active' widens the candidate pool so that "
            "sketch gate actually prunes in ablations; "
            "'node_summary_gate_multi' replaces mean-key sketches with "
            "multi-prototype block sketches. "
            "'simple' is a minimal explanation/debug path."
        ),
    )


def apply_mainline_profile(args, explicit_arg_dests=None):
    """Apply a profile, while preserving command-line overrides."""
    profile_name = getattr(args, "mainline_profile", "manual")
    profile = MAINLINE_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(f"Unknown mainline profile: {profile_name}")

    explicit_arg_dests = set(explicit_arg_dests or [])
    applied = {}
    for dest, value in profile.items():
        if dest in explicit_arg_dests:
            continue
        if hasattr(args, dest):
            setattr(args, dest, value)
            applied[dest] = value

    for dest, value in FORCED_MAINLINE_VALUES.items():
        if hasattr(args, dest):
            setattr(args, dest, value)
            applied[dest] = value

    args.applied_mainline_profile = profile_name
    args.applied_mainline_profile_values = applied
    return args


def collect_explicit_arg_dests(parser, argv):
    """Return argparse dest names explicitly supplied by the user/script."""
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit = set()
    for raw in argv:
        option = raw.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit.add(dest)
    return explicit
