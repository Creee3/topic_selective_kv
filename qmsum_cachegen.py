"""CacheGen comparison helpers for the QMSum mainline.

This module intentionally implements a conservative first baseline:
compress the full prompt KV once per document, estimate transfer TTFT from
the compressed byte count, and use the existing full-context answer F1 as the
quality proxy. It does not claim decompressed-KV answer quality.
"""

import math
import os
import sys
import time

import torch


def _ensure_local_lmcache_importable():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    lmcache_root = os.path.join(repo_dir, "lmcache")
    if lmcache_root not in sys.path:
        sys.path.insert(0, lmcache_root)


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _kv_3d_to_cachegen_blob(kv_3d, device="cuda:0"):
    """Convert HF per-layer 3D KV tuples to CacheGen's 5D HF tensor."""
    layers = []
    for key, value in kv_3d:
        key = key.detach().to(device)
        value = value.detach().to(device)
        layers.append(torch.stack((key, value), dim=0))
    return torch.stack(layers, dim=0).contiguous()


def _estimate_transfer_from_bytes(
    compressed_bytes,
    total_tokens,
    chunk_size,
    cost_config,
    prepare_ms,
    encode_ms,
    include_encode_time,
    cachegen_decode_ms,
    segment_count_mode,
):
    bandwidth_bytes_per_ms = max(
        1e-9,
        float(cost_config.get("fetch_bandwidth_bytes_per_ms", 1.0)),
    )
    per_node_rtt_ms = float(cost_config.get("per_node_rtt_ms", 0.0))
    per_segment_overhead_ms = float(cost_config.get("per_segment_overhead_ms", 0.0))
    decode_startup_ms = float(cost_config.get("decode_startup_ms", 0.0))
    kv_bytes_per_token = float(cost_config.get("kv_bytes_per_token", 0.0))

    cachegen_num_chunks = max(1, int(math.ceil(float(total_tokens) / max(1, chunk_size))))
    if segment_count_mode == "cachegen_chunks":
        transfer_segment_count = cachegen_num_chunks
    else:
        transfer_segment_count = 1

    raw_full_kv_bytes = float(total_tokens) * kv_bytes_per_token
    bandwidth_time_ms = float(compressed_bytes) / bandwidth_bytes_per_ms
    node_rtt_time_ms = per_node_rtt_ms
    segment_overhead_time_ms = float(transfer_segment_count) * per_segment_overhead_ms
    fetch_latency_ms = (
        bandwidth_time_ms + node_rtt_time_ms + segment_overhead_time_ms
    )
    total_encode_ms = float(prepare_ms) + float(encode_ms)
    ttft_no_encode_ms = (
        fetch_latency_ms + float(cachegen_decode_ms) + decode_startup_ms
    )
    ttft_with_encode_ms = total_encode_ms + ttft_no_encode_ms
    estimated_ttft_ms = (
        ttft_with_encode_ms if include_encode_time else ttft_no_encode_ms
    )

    compression_saving_ratio = 0.0
    if raw_full_kv_bytes > 0:
        compression_saving_ratio = 1.0 - (float(compressed_bytes) / raw_full_kv_bytes)

    return {
        "raw_full_kv_bytes": float(raw_full_kv_bytes),
        "raw_full_kv_mib": float(raw_full_kv_bytes / (1024.0 * 1024.0)),
        "compressed_bytes": float(compressed_bytes),
        "compressed_mib": float(compressed_bytes / (1024.0 * 1024.0)),
        "compression_saving_ratio": float(compression_saving_ratio),
        "cachegen_num_chunks": int(cachegen_num_chunks),
        "transfer_unit_count": 1,
        "transfer_segment_count": int(transfer_segment_count),
        "transfer_segment_mode": str(segment_count_mode),
        "bandwidth_time_ms": float(bandwidth_time_ms),
        "node_rtt_time_ms": float(node_rtt_time_ms),
        "segment_overhead_time_ms": float(segment_overhead_time_ms),
        "fetch_latency_ms": float(fetch_latency_ms),
        "prepare_ms": float(prepare_ms),
        "encode_ms": float(encode_ms),
        "total_encode_ms": float(total_encode_ms),
        "include_encode_time": bool(include_encode_time),
        "online_encode_ms": float(total_encode_ms if include_encode_time else 0.0),
        "cachegen_decode_ms": float(cachegen_decode_ms),
        "decode_startup_ms": float(decode_startup_ms),
        "estimated_ttft_ms": float(estimated_ttft_ms),
        "estimated_ttft_no_encode_ms": float(ttft_no_encode_ms),
        "estimated_ttft_with_encode_ms": float(ttft_with_encode_ms),
    }


def build_cachegen_full_estimate(
    kv_3d,
    total_tokens,
    cost_config,
    cachegen_model_name="mistral-community/Mistral-7B-v0.2",
    quant_level=2,
    chunk_size=256,
    include_encode_time=False,
    cachegen_decode_ms=0.0,
    segment_count_mode="one",
):
    """Compress full KV with CacheGen and return an estimated full-KV baseline."""
    prior_quant_level = os.environ.get("QUANT_LEVEL")
    try:
        _ensure_local_lmcache_importable()
        from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
        from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer

        os.environ["QUANT_LEVEL"] = str(int(quant_level))
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=int(chunk_size))
        metadata = LMCacheEngineMetadata(
            model_name=str(cachegen_model_name),
            fmt="huggingface",
            world_size=1,
            worker_id=0,
        )
        serializer = CacheGenSerializer(lmcache_config, metadata)

        prepare_start = time.perf_counter()
        kv_blob = _kv_3d_to_cachegen_blob(kv_3d)
        _cuda_sync()
        prepare_ms = 1000.0 * (time.perf_counter() - prepare_start)

        encode_start = time.perf_counter()
        compressed = serializer.to_bytes(kv_blob)
        _cuda_sync()
        encode_ms = 1000.0 * (time.perf_counter() - encode_start)
        compressed_bytes = len(compressed)

        del compressed, kv_blob
        torch.cuda.empty_cache()

        transfer_estimate = _estimate_transfer_from_bytes(
            compressed_bytes=compressed_bytes,
            total_tokens=total_tokens,
            chunk_size=int(chunk_size),
            cost_config=cost_config,
            prepare_ms=prepare_ms,
            encode_ms=encode_ms,
            include_encode_time=bool(include_encode_time),
            cachegen_decode_ms=float(cachegen_decode_ms),
            segment_count_mode=str(segment_count_mode),
        )
        transfer_estimate.update(
            {
                "enabled": True,
                "status": "ok",
                "baseline_type": "cachegen_full_estimated",
                "quality_note": (
                    "F1 uses full-context answer F1 as a proxy; this run does "
                    "not decode CacheGen KV for answer generation."
                ),
                "total_tokens": int(total_tokens),
                "cachegen_model_name": str(cachegen_model_name),
                "cachegen_quant_level": int(quant_level),
                "cachegen_chunk_size": int(chunk_size),
            }
        )
        return transfer_estimate
    except Exception as exc:
        torch.cuda.empty_cache()
        return {
            "enabled": True,
            "status": "error",
            "baseline_type": "cachegen_full_estimated",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "total_tokens": int(total_tokens),
            "cachegen_model_name": str(cachegen_model_name),
            "cachegen_quant_level": int(quant_level),
            "cachegen_chunk_size": int(chunk_size),
        }
    finally:
        if prior_quant_level is None:
            os.environ.pop("QUANT_LEVEL", None)
        else:
            os.environ["QUANT_LEVEL"] = prior_quant_level


def build_cachegen_case_metrics(cachegen_doc_estimate, system_cost, answer_eval=None):
    """Attach per-query selective-vs-CacheGen deltas to a doc-level estimate."""
    if not cachegen_doc_estimate:
        return None

    case_estimate = dict(cachegen_doc_estimate)
    if case_estimate.get("status") != "ok":
        return case_estimate

    selected_cost = (system_cost or {}).get("selected", {})
    selected_ttft_ms = float(selected_cost.get("estimated_ttft_ms", 0.0))
    cachegen_ttft_ms = float(case_estimate.get("estimated_ttft_ms", 0.0))
    case_estimate["selected_ttft_ms"] = float(selected_ttft_ms)
    case_estimate["selected_vs_cachegen_ttft_delta_ms"] = float(
        selected_ttft_ms - cachegen_ttft_ms
    )
    if cachegen_ttft_ms > 0:
        case_estimate["selected_vs_cachegen_ttft_saving_ratio"] = float(
            1.0 - (selected_ttft_ms / cachegen_ttft_ms)
        )
    else:
        case_estimate["selected_vs_cachegen_ttft_saving_ratio"] = 0.0

    if answer_eval:
        full_f1 = float(answer_eval.get("full_answer_f1", 0.0))
        selected_f1 = float(answer_eval.get("selected_answer_f1", 0.0))
        case_estimate["estimated_answer_f1"] = full_f1
        case_estimate["estimated_answer_f1_source"] = "full_answer_f1"
        case_estimate["selected_answer_f1"] = selected_f1
        case_estimate["selected_answer_f1_delta_vs_cachegen"] = float(
            selected_f1 - full_f1
        )
    return case_estimate
