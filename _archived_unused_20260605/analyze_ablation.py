"""
================================================================================
 analyze_ablation.py — 汇总消融实验结果到一张表
 用法: python analyze_ablation.py
================================================================================
"""
import json
import os
import glob

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "batch_longchat_*.json")))
    if not files:
        print("没有找到 LongChat 结果文件。请先运行 run_ablation.sh")
        return

    print(f"\n{'=' * 100}")
    print("LongChat 7B 消融实验结果汇总")
    print(f"{'=' * 100}\n")

    # 表头
    header = f"  {'实验组':<24} {'策略':<10} {'top_k':>5} {'chunk':>5} {'层':>4} {'n':>4} {'全量acc':>8} {'筛选acc':>8} {'节省%':>7} {'chunks':>6} {'MB':>7}"
    print(header)
    print(f"  {'-' * 90}")

    rows = []
    for fpath in files:
        data = load_json(fpath)
        cfg = data["config"]
        sm = data["summary"]

        strategy = cfg.get("per_head", False)
        threshold = cfg.get("adaptive_threshold", False)
        top_k = cfg.get("top_k", "-")
        chunk = cfg.get("chunk_size", 256)
        first = cfg.get("first", 0)
        last = cfg.get("last", 0)

        if strategy and not threshold:
            strat_name = "per_head"
        elif threshold:
            strat_name = f"threshold"
        else:
            strat_name = "mean"

        # 推断层数: 从文件名或summary
        fname = os.path.basename(fpath)
        if "5l" in fname:
            layers = "5"
        elif "1l" in fname:
            layers = "1"
        else:
            layers = "?"

        n = sm.get("n", "?")
        full_acc = sm.get("full_acc", sm.get("avg_full_acc", "-"))
        sel_acc = sm.get("sel_acc", sm.get("avg_sel_acc", "-"))

        if isinstance(full_acc, str):
            full_disp = full_acc
            sel_disp = sel_acc
        elif isinstance(full_acc, float):
            full_disp = f"{full_acc:.3f}"
            sel_disp = f"{sel_acc:.3f}"
        else:
            full_disp = str(full_acc)
            sel_disp = str(sel_acc)

        save_pct = sm.get("avg_token_savings_pct", 0)
        avg_chunks = sm.get("avg_chunks", 0)
        avg_mb = sm.get("avg_selected_mb", 0)

        # 解析准确率字符串
        full_str = f"{full_disp}"
        sel_str = f"{sel_disp}"

        print(f"  {fname:<24} {strat_name:<10} {str(top_k):>5} {str(chunk):>5} {layers:>4} "
              f"{str(n):>4} {full_str:>8} {sel_str:>8} {save_pct:>6.1f}% {avg_chunks:>5.1f} {avg_mb:>6.1f}")

        rows.append({
            "file": fname, "strategy": strat_name, "top_k": top_k,
            "chunk": chunk, "layers": layers, "n": n,
            "full_acc": full_acc, "sel_acc": sel_acc,
            "save": save_pct, "chunks": avg_chunks, "mb": avg_mb,
        })

    # ---- 分组对比 ----
    if rows:
        print(f"\n{'=' * 100}")
        print("关键发现")
        print(f"{'=' * 100}")

        # 1. top_k 对比 (固定 per_head, c256, 5l)
        topk_rows = [r for r in rows if r["strategy"] == "per_head" and r["chunk"] == 256 and r["layers"] == "5"]
        if len(topk_rows) >= 2:
            print(f"\n  1. top_k 扫描 (per_head, c256, 5层):")
            for r in sorted(topk_rows, key=lambda x: x["top_k"]):
                acc_str = r["sel_acc"] if isinstance(r["sel_acc"], str) else f"{r['sel_acc']:.3f}"
                print(f"     top_k={r['top_k']}: acc={acc_str}, save={r['save']:.0f}%, chunks={r['chunks']:.1f}")

        # 2. 策略对比 (固定 k=2, c256, 5l)
        strat_rows = [r for r in rows if r["top_k"] == 2 and r["chunk"] == 256 and r["layers"] == "5"]
        if len(strat_rows) >= 2:
            print(f"\n  2. 打分策略对比 (top_k=2, c256, 5层):")
            for r in sorted(strat_rows, key=lambda x: x["strategy"]):
                acc_str = r["sel_acc"] if isinstance(r["sel_acc"], str) else f"{r['sel_acc']:.3f}"
                print(f"     {r['strategy']:<12}: acc={acc_str}, save={r['save']:.0f}%, chunks={r['chunks']:.1f}")

        # 3. chunk 对比 (固定 per_head, k=2, 5l)
        chunk_rows = [r for r in rows if r["strategy"] == "per_head" and r["top_k"] == 2 and r["layers"] == "5"]
        if len(chunk_rows) >= 2:
            print(f"\n  3. chunk_size 对比 (per_head, top_k=2, 5层):")
            for r in sorted(chunk_rows, key=lambda x: x["chunk"]):
                acc_str = r["sel_acc"] if isinstance(r["sel_acc"], str) else f"{r['sel_acc']:.3f}"
                print(f"     c{r['chunk']}: acc={acc_str}, save={r['save']:.0f}%, chunks={r['chunks']:.1f}, MB={r['mb']:.1f}")

        # 4. 层数对比 (固定 per_head, k=2, c256)
        layer_rows = [r for r in rows if r["strategy"] == "per_head" and r["top_k"] == 2 and r["chunk"] == 256]
        if len(layer_rows) >= 2:
            print(f"\n  4. 打分层的消融 (per_head, top_k=2, c256):")
            for r in sorted(layer_rows, key=lambda x: x["layers"]):
                acc_str = r["sel_acc"] if isinstance(r["sel_acc"], str) else f"{r['sel_acc']:.3f}"
                print(f"     {r['layers']}层: acc={acc_str}, save={r['save']:.0f}%, chunks={r['chunks']:.1f}")

    print()


if __name__ == "__main__":
    main()
