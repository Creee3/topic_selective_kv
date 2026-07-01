"""CacheGen comparison helpers for the QMSum mainline."""

import math
import os
import sys
import time

import torch

from qmsum_answering import (
    build_answer_retry_prompt,
    compute_text_f1,
    detect_bad_answer_output,
    postprocess_generated_answer_text,
)
from qmsum_data import build_qmsum_answer_prompt


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


def _cachegen_blob_to_hf_past(kv_blob, upto_tokens=None):
    """Convert CacheGen's HuggingFace 5D tensor to HF legacy past_key_values."""
    if kv_blob.dim() != 5 or kv_blob.shape[1] != 2:
        raise ValueError(
            "Expected CacheGen HuggingFace blob shape "
            "(layers, 2, heads, tokens, head_dim), got "
            f"{tuple(kv_blob.shape)}"
        )
    if upto_tokens is not None:
        kv_blob = kv_blob[:, :, :, : int(upto_tokens), :]

    past = []
    for layer_idx in range(kv_blob.shape[0]):
        key = kv_blob[layer_idx, 0].unsqueeze(0).contiguous()
        value = kv_blob[layer_idx, 1].unsqueeze(0).contiguous()
        past.append((key, value))
    return tuple(past)


def _build_cachegen_codecs(cachegen_model_name, quant_level, chunk_size):
    _ensure_local_lmcache_importable()
    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
    from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
    from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer

    os.environ["QUANT_LEVEL"] = str(int(quant_level))
    lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=int(chunk_size))
    metadata = LMCacheEngineMetadata(
        model_name=str(cachegen_model_name),
        fmt="huggingface",
        world_size=1,
        worker_id=0,
    )
    return (
        CacheGenSerializer(lmcache_config, metadata),
        CacheGenDeserializer(lmcache_config, metadata),
    )


def _greedy_generate_from_past(
    model,
    tokenizer,
    prompt_input_ids,
    prompt_past,
    max_new_tokens,
):
    """Generate greedily from a prompt KV by recomputing only the last token."""
    if prompt_input_ids.shape[1] < 2:
        raise ValueError("CacheGen roundtrip prompt must contain at least two tokens")

    # Use decompressed KV for tokens [0, T-1), then feed the last prompt token.
    # This avoids duplicating the last token in the cache while still testing the
    # decompressed context for the first generated token.
    past = _cachegen_blob_to_hf_past(
        prompt_past,
        upto_tokens=prompt_input_ids.shape[1] - 1,
    )
    step_input = prompt_input_ids[:, -1:]
    generated_tokens = []

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            outputs = model.model(
                input_ids=step_input,
                past_key_values=past,
                use_cache=True,
            )
            logits = model.lm_head(outputs[0])
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated_tokens.append(next_token)
            past = outputs.past_key_values
            step_input = next_token
            if next_token.item() == tokenizer.eos_token_id:
                break

    if generated_tokens:
        generated_ids = torch.cat(generated_tokens, dim=-1)
    else:
        generated_ids = prompt_input_ids.new_empty((prompt_input_ids.shape[0], 0))
    return generated_ids


def _roundtrip_generate_once(
    model,
    tokenizer,
    prompt_text,
    max_new_tokens,
    cachegen_model_name,
    quant_level,
    chunk_size,
    cost_config,
    include_encode_time,
    segment_count_mode,
):
    tokenize_start = time.perf_counter()
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    _cuda_sync()
    tokenize_ms = 1000.0 * (time.perf_counter() - tokenize_start)
    total_tokens = int(input_ids.shape[1])

    # The vendored CacheGen decoder uses config.chunk_size as its output
    # buffer capacity. Keep the user-facing chunk_size for transfer accounting,
    # but give the decoder enough room for the full answer prompt.
    codec_buffer_tokens = max(int(chunk_size), total_tokens)
    serializer, deserializer = _build_cachegen_codecs(
        cachegen_model_name=cachegen_model_name,
        quant_level=quant_level,
        chunk_size=codec_buffer_tokens,
    )

    prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, use_cache=True)
    _cuda_sync()
    prefill_ms = 1000.0 * (time.perf_counter() - prefill_start)

    kv_3d = tuple((layer[0][0], layer[1][0]) for layer in outputs.past_key_values)
    prepare_start = time.perf_counter()
    kv_blob = _kv_3d_to_cachegen_blob(kv_3d)
    _cuda_sync()
    prepare_ms = 1000.0 * (time.perf_counter() - prepare_start)

    encode_start = time.perf_counter()
    compressed = serializer.to_bytes(kv_blob)
    _cuda_sync()
    encode_ms = 1000.0 * (time.perf_counter() - encode_start)
    compressed_bytes = len(compressed)

    decode_start = time.perf_counter()
    decoded_blob = deserializer.from_bytes(compressed)
    _cuda_sync()
    decode_ms = 1000.0 * (time.perf_counter() - decode_start)

    generate_start = time.perf_counter()
    generated_ids = _greedy_generate_from_past(
        model,
        tokenizer,
        input_ids,
        decoded_blob,
        max_new_tokens=max_new_tokens,
    )
    _cuda_sync()
    generation_ms = 1000.0 * (time.perf_counter() - generate_start)

    raw_answer_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
    answer_text, postprocess_actions = postprocess_generated_answer_text(
        raw_answer_text
    )

    transfer_estimate = _estimate_transfer_from_bytes(
        compressed_bytes=compressed_bytes,
        total_tokens=total_tokens,
        chunk_size=int(chunk_size),
        cost_config=cost_config,
        prepare_ms=prepare_ms,
        encode_ms=encode_ms,
        include_encode_time=bool(include_encode_time),
        cachegen_decode_ms=float(decode_ms),
        segment_count_mode=str(segment_count_mode),
    )
    transfer_estimate.update(
        {
            "prompt_tokens": int(total_tokens),
            "tokenize_ms": float(tokenize_ms),
            "prefill_ms": float(prefill_ms),
            "measured_decode_ms": float(decode_ms),
            "generation_ms": float(generation_ms),
            "raw_answer_text": raw_answer_text,
            "answer": answer_text,
            "postprocess_actions": postprocess_actions,
        }
    )

    del generated_ids, decoded_blob, compressed, kv_blob, kv_3d, outputs, input_ids
    torch.cuda.empty_cache()
    return transfer_estimate


def build_cachegen_roundtrip_answer_eval(
    model,
    tokenizer,
    transcripts,
    query_text,
    gold_answer,
    max_new_tokens,
    cost_config,
    answer_prompt_style="basic",
    cachegen_model_name="mistral-community/Mistral-7B-v0.2",
    quant_level=2,
    chunk_size=256,
    include_encode_time=False,
    segment_count_mode="one",
    answer_eval=None,
    system_cost=None,
):
    """Compress, decompress, generate from the decoded KV, and compute answer F1."""
    prior_quant_level = os.environ.get("QUANT_LEVEL")
    try:
        full_prompt = build_qmsum_answer_prompt(
            transcripts,
            query_text,
            turn_indices=None,
            prompt_style=answer_prompt_style,
        )
        first = _roundtrip_generate_once(
            model=model,
            tokenizer=tokenizer,
            prompt_text=full_prompt,
            max_new_tokens=max_new_tokens,
            cachegen_model_name=cachegen_model_name,
            quant_level=quant_level,
            chunk_size=chunk_size,
            cost_config=cost_config,
            include_encode_time=include_encode_time,
            segment_count_mode=segment_count_mode,
        )
        initial_bad = detect_bad_answer_output(first["answer"])
        used = first
        retry = None
        retried = False
        used_retry = False

        if initial_bad["is_bad"]:
            retried = True
            retry_prompt = build_answer_retry_prompt(full_prompt)
            retry = _roundtrip_generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt_text=retry_prompt,
                max_new_tokens=max_new_tokens,
                cachegen_model_name=cachegen_model_name,
                quant_level=quant_level,
                chunk_size=chunk_size,
                cost_config=cost_config,
                include_encode_time=include_encode_time,
                segment_count_mode=segment_count_mode,
            )
            retry_bad = detect_bad_answer_output(retry["answer"])
            if not retry_bad["is_bad"]:
                used = retry
                used_retry = True
        else:
            retry_bad = {"is_bad": False, "reasons": []}

        final_bad = detect_bad_answer_output(used["answer"])
        answer_f1 = compute_text_f1(used["answer"], gold_answer)

        result = dict(used)
        result.update(
            {
                "enabled": True,
                "status": "ok",
                "baseline_type": "cachegen_full_roundtrip_answer",
                "quality_note": (
                    "F1 is generated from a CacheGen-compressed then "
                    "decompressed full answer prompt KV."
                ),
                "cachegen_model_name": str(cachegen_model_name),
                "cachegen_quant_level": int(quant_level),
                "cachegen_chunk_size": int(chunk_size),
                "answer_prompt_style": str(answer_prompt_style),
                "max_new_tokens": int(max_new_tokens),
                "gold_answer": gold_answer,
                "answer_f1": float(answer_f1),
                "bad_output": bool(final_bad["is_bad"]),
                "bad_output_reasons": final_bad["reasons"],
                "initial_bad_output": bool(initial_bad["is_bad"]),
                "initial_bad_output_reasons": initial_bad["reasons"],
                "retried": bool(retried),
                "used_retry": bool(used_retry),
                "retry_bad_output": bool(retry_bad["is_bad"]),
                "retry_bad_output_reasons": retry_bad["reasons"],
            }
        )
        if retry is not None:
            result["retry_answer"] = retry.get("answer", "")
            result["retry_answer_f1"] = float(
                compute_text_f1(retry.get("answer", ""), gold_answer)
            )
            result["retry_postprocess_actions"] = retry.get(
                "postprocess_actions",
                [],
            )

        if answer_eval:
            full_f1 = float(answer_eval.get("full_answer_f1", 0.0))
            selected_f1 = float(answer_eval.get("selected_answer_f1", 0.0))
            result["full_answer_f1"] = float(full_f1)
            result["selected_answer_f1"] = float(selected_f1)
            result["answer_f1_delta_vs_full"] = float(answer_f1 - full_f1)
            result["answer_f1_delta_vs_selected"] = float(answer_f1 - selected_f1)
            result["selected_answer_f1_delta_vs_cachegen_roundtrip"] = float(
                selected_f1 - answer_f1
            )

        selected_cost = (system_cost or {}).get("selected", {})
        selected_ttft_ms = float(selected_cost.get("estimated_ttft_ms", 0.0))
        cachegen_ttft_ms = float(result.get("estimated_ttft_ms", 0.0))
        result["selected_ttft_ms"] = float(selected_ttft_ms)
        result["selected_vs_cachegen_roundtrip_ttft_delta_ms"] = float(
            selected_ttft_ms - cachegen_ttft_ms
        )
        if cachegen_ttft_ms > 0:
            result["selected_vs_cachegen_roundtrip_ttft_saving_ratio"] = float(
                1.0 - (selected_ttft_ms / cachegen_ttft_ms)
            )
        else:
            result["selected_vs_cachegen_roundtrip_ttft_saving_ratio"] = 0.0
        return result
    except Exception as exc:
        torch.cuda.empty_cache()
        return {
            "enabled": True,
            "status": "error",
            "baseline_type": "cachegen_full_roundtrip_answer",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "cachegen_model_name": str(cachegen_model_name),
            "cachegen_quant_level": int(quant_level),
            "cachegen_chunk_size": int(chunk_size),
        }
    finally:
        if prior_quant_level is None:
            os.environ.pop("QUANT_LEVEL", None)
        else:
            os.environ["QUANT_LEVEL"] = prior_quant_level


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
