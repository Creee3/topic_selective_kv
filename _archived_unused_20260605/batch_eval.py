"""
================================================================================
 batch_eval.py — 批量评测 per_head 选择性 KV 传输

 支持数据集：longchat, tqa (TriviaQA), nqa (NarrativeQA)

 运行：
   # LongChat
   python batch_eval.py --dataset longchat --model_id ~/models/mistral-7b/ \
       --num_gpus 1 --max_gpu_memory 40 --per_head --top_k 2

   # TriviaQA (natural language QA)
   python batch_eval.py --dataset tqa --model_id ~/models/mistral-7b/ \
       --num_gpus 1 --max_gpu_memory 40 --per_head --top_k 2
================================================================================
"""

import sys
import os
import argparse
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ORDINAL_TO_NUMBER = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
}
NUMBER_TO_ORDINAL = {v: k for k, v in ORDINAL_TO_NUMBER.items() if not k[0].isdigit()}


def load_sample(filepath, doc_id):
    """加载一条样本，兼容 longchat / tqa / nqa 三种格式"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == doc_id:
                return json.loads(line.strip())
    raise IndexError(f"doc_id={doc_id} 超出文件行数范围")


def extract_query(prompt, dataset):
    """从 prompt 中提取用户询问部分，用于 attention 打分"""
    if dataset == "longchat":
        if "Now the record ends." in prompt:
            return prompt.split("Now the record ends.")[-1].strip()
        return prompt[prompt.rfind("USER:"):].split("ASSISTANT:")[0].strip()
    else:
        # TQA/NQA: prompt 末尾有 "Question: xxx"
        if "Question:" in prompt:
            return prompt.split("Question:")[-1].strip()
        return prompt[-500:]  # fallback: 最后 500 字符


def main(args):
    print("=" * 70)
    print(f"批量评测: dataset={args.dataset}, per_head={args.per_head}")
    print(f"  top_k={args.top_k}, keep_first={args.keep_first_chunks}, keep_last={args.keep_last_chunks}")
    print("=" * 70)

    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
    from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
    from src.utils import (
        define_model_and_tokenizer, to_blob, split_kv, calculate_acc,
    )
    from experiment_chunk_split import _chunk_qk_scores_per_head

    # ---- 1. 加载模型（只一次） ----
    print("加载模型...")
    model, tokenizer = define_model_and_tokenizer(
        args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory,
    )
    print("模型加载完毕\n")

    num_model_layers = len(model.model.layers)
    if args.scoring_layers:
        scoring_layers = [int(x) for x in args.scoring_layers.split(",")]
    else:
        scoring_layers = [0, 8, 16, 24, num_model_layers - 1]
    scoring_layers = [l if l >= 0 else num_model_layers + l for l in scoring_layers]

    chunk_size = args.chunk_size
    first_n = args.keep_first_chunks
    last_n = args.keep_last_chunks
    cg_model_name = args.cachegen_model_name

    results = []
    t_start = time.perf_counter()

    for doc_id in range(args.start_doc, args.end_doc):
        try:
            data = load_sample(args.data_path, doc_id=doc_id)
        except IndexError:
            break

        prompt = data['prompt']
        full_query = extract_query(prompt, args.dataset)

        # ---- 序数词替换（测试不同位置 topic 的检索能力） ----
        if args.target_ordinal and args.dataset == "longchat":
            target_num = ORDINAL_TO_NUMBER.get(args.target_ordinal.lower())
            if target_num is None:
                try:
                    target_num = int(args.target_ordinal)
                except ValueError:
                    print(f"⚠ 无法解析序数词: {args.target_ordinal}，跳过替换")
                    target_num = None
            if target_num is not None:
                target_word = NUMBER_TO_ORDINAL.get(target_num, args.target_ordinal)
                original_word = None
                for word in ORDINAL_TO_NUMBER:
                    if word in full_query.lower():
                        original_word = word
                        break
                if original_word:
                    full_query = full_query.replace(original_word, target_word)
                    prompt = prompt.replace(
                        f"the {original_word} topic", f"the {target_word} topic"
                    )

        # ---- Prefill ----
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, use_cache=True)
        kv_3d = tuple((layer[0][0], layer[1][0]) for layer in outputs.past_key_values)
        kv_4d = outputs.past_key_values  # (1, heads, seq, dim) — 推理直接用
        total_tokens = kv_3d[0][0].shape[1]

        # ---- 切 chunk + 提取 K ----
        num_chunks = total_tokens // chunk_size
        if total_tokens % chunk_size != 0:
            num_chunks += 1

        chunk_keys_list = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_tokens)
            kv_chunk = split_kv(kv_3d, start, end)
            k_stacked = torch.stack([layer[0].squeeze(0) for layer in kv_chunk], dim=0)
            chunk_keys_list.append(k_stacked)

        # ---- 选择 chunk ----
        if first_n + last_n >= num_chunks:
            selected_indices = list(range(num_chunks))
            selected_middle_count = 0
        else:
            middle_indices = list(range(first_n, num_chunks - last_n))
            n_middle = len(middle_indices)
            middle_keys = [chunk_keys_list[i] for i in middle_indices]

            query_ids = tokenizer(full_query, return_tensors="pt").input_ids.cuda()
            scores_matrix, _ = _chunk_qk_scores_per_head(
                query_ids, middle_keys, model, scoring_layers
            )
            n_heads = scores_matrix.shape[1]

            if args.per_head:
                selected_middle = set()
                k_per_head = max(1, args.top_k)
                for h in range(n_heads):
                    head_scores = scores_matrix[:, h]
                    top_k_local = np.argsort(head_scores)[-k_per_head:][::-1]
                    selected_middle.update(middle_indices[j] for j in top_k_local)
                selected_middle = sorted(selected_middle)
            elif args.adaptive_threshold:
                mean_scores = scores_matrix.mean(axis=1)
                score_range = mean_scores.max() - mean_scores.min()
                threshold = mean_scores.max() - args.alpha * score_range
                selected_middle = sorted([
                    middle_indices[j] for j, s in enumerate(mean_scores)
                    if s >= threshold
                ])
            else:
                mean_scores = scores_matrix.mean(axis=1)
                k = min(args.top_k, n_middle)
                top_k_local = np.argsort(mean_scores)[-k:][::-1]
                selected_middle = sorted([middle_indices[j] for j in top_k_local])

            selected_middle_count = len(selected_middle)
            selected_indices = (
                list(range(first_n)) + selected_middle +
                list(range(num_chunks - last_n, num_chunks))
            )
            selected_indices = sorted(set(selected_indices))

            del scores_matrix, middle_keys, query_ids
            torch.cuda.empty_cache()

        # ---- 构建 selected_input_ids + Re-prefill ----
        selected_parts = []
        for i in selected_indices:
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_tokens)
            selected_parts.append(input_ids[0, start:end])
        selected_input_ids = torch.cat(selected_parts, dim=0).unsqueeze(0)

        with torch.no_grad():
            sel_outputs = model.model(input_ids=selected_input_ids, use_cache=True)
        kv_sel_4d = sel_outputs.past_key_values
        kv_sel_3d = tuple((l[0][0], l[1][0]) for l in kv_sel_4d)
        selected_tokens = kv_sel_3d[0][0].shape[1]
        token_savings = 100 * (1 - selected_tokens / total_tokens)

        # ---- CacheGen 压缩（只测 selected） ----
        kv_sel_tensor = to_blob(kv_sel_3d)
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
        meta_data = LMCacheEngineMetadata(
            model_name=cg_model_name, fmt="huggingface", world_size=1, worker_id=0
        )
        os.environ["QUANT_LEVEL"] = str(args.quant_level)
        serializer = CacheGenSerializer(lmcache_config, meta_data)
        selected_bytes = serializer.to_bytes(kv_sel_tensor)
        torch.cuda.synchronize()
        selected_mb = len(selected_bytes) / 1e6
        del kv_sel_tensor
        torch.cuda.empty_cache()

        # ---- 推理: 全量 + 筛选 ----
        with torch.no_grad():
            full_gen = input_ids
            past_kv = kv_4d
            for _ in range(20):
                outputs = model.model(input_ids=full_gen[:, -1:], past_key_values=past_kv, use_cache=True)
                logits = model.lm_head(outputs[0])
                next_token = logits[:, -1:, :].argmax(dim=-1)
                full_gen = torch.cat([full_gen, next_token], dim=-1)
                past_kv = outputs.past_key_values
                if next_token.item() == tokenizer.eos_token_id:
                    break
        full_pred = tokenizer.decode(full_gen[0][input_ids.shape[1]:], skip_special_tokens=True)
        del past_kv

        with torch.no_grad():
            sel_gen = selected_input_ids
            past_kv = kv_sel_4d
            for _ in range(20):
                outputs = model.model(input_ids=sel_gen[:, -1:], past_key_values=past_kv, use_cache=True)
                logits = model.lm_head(outputs[0])
                next_token = logits[:, -1:, :].argmax(dim=-1)
                sel_gen = torch.cat([sel_gen, next_token], dim=-1)
                past_kv = outputs.past_key_values
                if next_token.item() == tokenizer.eos_token_id:
                    break
        selected_pred = tokenizer.decode(sel_gen[0][selected_input_ids.shape[1]:], skip_special_tokens=True)
        del past_kv

        # ---- 准确率 ----
        if args.dataset == "longchat":
            # data['label'] 是全部 15 个 topic 的列表
            # 默认 query 问 "first topic"，对应 label[0]
            # target_ordinal 改变时，需要对应调整 label 索引
            if args.target_ordinal:
                target_num = ORDINAL_TO_NUMBER.get(args.target_ordinal.lower())
                if target_num is None:
                    try:
                        target_num = int(args.target_ordinal)
                    except ValueError:
                        target_num = 1
                label_idx = target_num - 1
            else:
                label_idx = 0
            # 取单个 topic string，calculate_acc 用 label[0] 取到它
            label_or_data = [data['label'][label_idx]]
        else:
            label_or_data = data
        full_acc = calculate_acc(args.dataset, full_pred, label_or_data)
        selected_acc = calculate_acc(args.dataset, selected_pred, label_or_data)
        pred_match = (full_pred.strip() == selected_pred.strip())

        # 打印预测
        answer_hint = str(data.get('answers', data.get('label', '')))
        if args.dataset == "longchat" and isinstance(data.get('label'), list):
            answer_hint = str(data['label'][label_idx])
        match_flag = "✓" if pred_match else "✗"
        ordinal_hint = f"[topic={args.target_ordinal}]" if args.target_ordinal else ""
        print(f"[{doc_id:>2}] {ordinal_hint} 全量:{total_tokens}→筛选:{selected_tokens}(+{selected_middle_count}chk) "
              f"节省:{token_savings:.0f}% {selected_mb:.1f}MB "
              f"全量F1:{full_acc:.3f} 筛选F1:{selected_acc:.3f} 一致:{match_flag}")
        print(f"    全量 pred: \"{full_pred[:120]}\"")
        print(f"    筛选 pred: \"{selected_pred[:120]}\"")
        print(f"    参考答案:  \"{answer_hint[:120]}\"")

        results.append({
            "doc_id": doc_id,
            "full_tokens": total_tokens,
            "selected_tokens": selected_tokens,
            "selected_chunks": len(selected_indices),
            "middle_chunks_selected": selected_middle_count,
            "token_savings_pct": token_savings,
            "selected_size_mb": selected_mb,
            "full_accuracy": full_acc,
            "selected_accuracy": selected_acc,
            "prediction_match": pred_match,
        })

        del input_ids, selected_input_ids, full_gen, sel_gen
        del kv_3d, kv_4d, kv_sel_3d, kv_sel_4d, chunk_keys_list
        torch.cuda.empty_cache()

    # ---- 汇总 ----
    n = len(results)
    if n == 0:
        print("无结果"); return

    avg_token_save = np.mean([r["token_savings_pct"] for r in results])
    avg_chunks = np.mean([r["selected_chunks"] for r in results])
    avg_mb = np.mean([r["selected_size_mb"] for r in results])
    avg_sel_tokens = np.mean([r["selected_tokens"] for r in results])
    avg_full_acc = np.mean([r["full_accuracy"] for r in results])
    avg_sel_acc = np.mean([r["selected_accuracy"] for r in results])
    if "prediction_match" in results[0]:
        match_count = sum(1 for r in results if r["prediction_match"])
        match_rate = match_count / n

    is_longchat = (args.dataset == "longchat")
    if is_longchat:
        full_correct = sum(1 for r in results if r["full_accuracy"] == 1)
        sel_correct = sum(1 for r in results if r["selected_accuracy"] == 1)
        sel_worse = sum(1 for r in results if r["full_accuracy"] == 1 and r["selected_accuracy"] == 0)

    print(f"\n{'=' * 70}")
    print(f"汇总 (n={n})")
    print(f"{'=' * 70}")
    print(f"  平均全量 tokens:  {np.mean([r['full_tokens'] for r in results]):.0f}")
    print(f"  平均筛选 tokens:  {avg_sel_tokens:.0f}")
    print(f"  平均选中 chunk:   {avg_chunks:.1f}")
    print(f"  平均 token 节省:  {avg_token_save:.1f}%")
    print(f"  平均压缩大小:     {avg_mb:.1f} MB")
    if is_longchat:
        print(f"  全量准确率:       {full_correct}/{n} ({100*full_correct/n:.1f}%)")
        print(f"  筛选准确率:       {sel_correct}/{n} ({100*sel_correct/n:.1f}%)")
        print(f"  筛选退化:         {sel_worse}/{n}")
    else:
        print(f"  平均全量 F1:      {avg_full_acc:.3f}")
        print(f"  平均筛选 F1:      {avg_sel_acc:.3f}")
        print(f"  F1 下降:          {avg_full_acc - avg_sel_acc:.3f}")
    if "prediction_match" in results[0]:
        print(f"  全量vs筛选一致:   {match_count}/{n} ({100*match_rate:.1f}%)")

    strategy = "ph" if args.per_head else ("th" if args.adaptive_threshold else "mean")
    layers_tag = f"{len(scoring_layers)}l" if args.scoring_layers else "5l"
    ordinal_tag = f"_ord{args.target_ordinal}" if args.target_ordinal else ""
    out_path = f"outputs/batch_{args.dataset}_{strategy}_k{args.top_k}_c{chunk_size}_{layers_tag}{ordinal_tag}.json"
    os.makedirs("outputs", exist_ok=True)
    summary_data = {
        "n": n, "avg_token_savings_pct": avg_token_save,
        "avg_selected_mb": avg_mb, "avg_chunks": avg_chunks,
        "avg_full_acc": avg_full_acc,
        "avg_sel_acc": avg_sel_acc,
    }
    if is_longchat:
        summary_data["full_acc"] = f"{full_correct}/{n}"
        summary_data["sel_acc"] = f"{sel_correct}/{n}"
        summary_data["sel_worse"] = sel_worse
    if "prediction_match" in results[0]:
        summary_data["prediction_match_rate"] = match_rate
    with open(out_path, "w") as f:
        json.dump({
            "config": {"top_k": args.top_k, "per_head": args.per_head,
                       "first": first_n, "last": last_n, "chunk_size": chunk_size,
                       "target_ordinal": args.target_ordinal},
            "summary": summary_data,
            "details": results,
        }, f, indent=2)
    print(f"\n保存到 {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="批量评测选择性 KV 传输")
    p.add_argument("--dataset", type=str, default="longchat",
                   choices=["longchat", "tqa", "nqa"],
                   help="数据集 (longchat/tqa/nqa)")
    p.add_argument("--data_path", type=str, default=None,
                   help="数据文件路径（默认根据 --dataset 自动选择）")
    p.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_gpu_memory", type=int, default=48)
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--keep_first_chunks", type=int, default=0)
    p.add_argument("--keep_last_chunks", type=int, default=0)
    p.add_argument("--target_ordinal", type=str, default=None,
                   help="覆盖 query 中的序数词 (如 first/third/eighth)，测试不同位置 topic")
    p.add_argument("--per_head", action="store_true", default=False)
    p.add_argument("--adaptive_threshold", action="store_true", default=False)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--scoring_layers", type=str, default=None)
    p.add_argument("--quant_level", type=int, default=2)
    p.add_argument("--cachegen_model_name", type=str,
                   default="mistral-community/Mistral-7B-v0.2")
    p.add_argument("--start_doc", type=int, default=0)
    p.add_argument("--end_doc", type=int, default=50,
                   help="跑多少条 (longchat 默认50, tqa/nqa 建议先试10)")

    args = p.parse_args()

    # 自动推断数据路径
    if args.data_path is None:
        base = os.path.join(os.path.dirname(__file__), "..", "CacheGen-main", "test_data")
        path_map = {"longchat": "longchat.jsonl", "tqa": "tqa.jsonl", "nqa": "nqa.jsonl"}
        args.data_path = os.path.abspath(os.path.join(base, path_map[args.dataset]))

    if not os.path.exists(args.data_path):
        print(f"数据文件不存在: {args.data_path}")
        print("请用 --data_path 指定路径")
        sys.exit(1)

    print(f"数据: {args.data_path} ({args.dataset})")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    main(args)
