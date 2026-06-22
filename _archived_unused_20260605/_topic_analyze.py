"""多位置 topic 测试结果分析"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "outputs")

baseline_file = os.path.join(OUT, "batch_longchat_ph_k2_c256_5l.json")
ord_files = {
    "first": os.path.join(OUT, "batch_longchat_ph_k2_c256_5l_ordfirst.json"),
    "third": os.path.join(OUT, "batch_longchat_ph_k2_c256_5l_ordthird.json"),
    "fifth": os.path.join(OUT, "batch_longchat_ph_k2_c256_5l_ordfifth.json"),
}

print("=" * 90)
print("LongChat 多位置 topic 检索测试")
print("  per_head, top_k=2, c256, 5 layers")
print("=" * 90)

# Header
print(f"\n{'Topic':<14} {'筛选acc':>10} {'节省%':>8} {'chunks':>8} {'MB':>8} {'预测一致率':>12}")
print("-" * 60)

all_data = {}

# Load baseline
with open(baseline_file) as f:
    all_data["末尾(基线)"] = json.load(f)

for label, fname in ord_files.items():
    with open(fname) as f:
        all_data[label] = json.load(f)

for label, data in all_data.items():
    sm = data["summary"]
    sel_acc = sm.get("sel_acc", sm.get("avg_sel_acc", ""))
    save = sm["avg_token_savings_pct"]
    chunks = sm["avg_chunks"]
    mb = sm["avg_selected_mb"]
    match_rate = sm.get("prediction_match_rate", None)

    acc_str = str(sel_acc) if isinstance(sel_acc, str) else f"{sel_acc:.3f}"
    match_str = f"{100*match_rate:.0f}%" if match_rate is not None else "N/A"
    print(f"{label:<14} {acc_str:>10} {save:>7.1f}% {chunks:>7.1f} {mb:>7.1f} {match_str:>12}")

# 2. Detailed comparison: which samples fail in each position
print()
print("=" * 90)
print("2. prediction_match=0 的样本")
print("=" * 90)
for label in ["first", "third", "fifth"]:
    data = all_data[label]
    mismatch = [d["doc_id"] for d in data["details"] if not d["prediction_match"]]
    match_rate = data["summary"]["prediction_match_rate"]
    print(f"\n  {label} topic: 不一致率={100*(1-match_rate):.0f}%, 不一致样本={mismatch}")

# 3. Show a few specific examples for inspection
print()
print("=" * 90)
print("3. 样例抽查 (third topic, doc_id=0)")
print("=" * 90)
data = all_data["third"]
for d in data["details"]:
    if d["doc_id"] == 0:
        print(f"  chunks selected: {d['selected_chunks']}, middle: {d['middle_chunks_selected']}")
        print(f"  token savings: {d['token_savings_pct']:.0f}%")
        print(f"  prediction_match: {d['prediction_match']}")
        break

# 4. Summary
print()
print("=" * 90)
print("4. 汇总")
print("=" * 90)
for label, data in all_data.items():
    sm = data["summary"]
    match_rate = sm.get("prediction_match_rate", None)
    sel_acc = sm.get("sel_acc", sm.get("avg_sel_acc", ""))
    n = sm["n"]
    print(f"  {label:<14} n={n}, label-acc={sel_acc}, consistency={100*match_rate:.0f}%" if match_rate else f"  {label:<14} n={n}, label-acc={sel_acc}")
