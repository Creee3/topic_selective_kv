"""
Central presets for the QMSum selective-KV mainline.

The project has accumulated many ablation switches.  Keep those switches
available, but make the default "current" path explicit and easy to audit.
"""


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
            "defaults, 'current' is the latest stable selective-fetch path, "
            "'quality_guard' is a conservative candidate-recall path, and "
            "'turn_rerank' tests turn-level final evidence selection. "
            "'hybrid_select' keeps a raw-QK core plus turn-level coverage. "
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
