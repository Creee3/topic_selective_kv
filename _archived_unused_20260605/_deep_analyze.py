"""深入分析消融实验结果"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "outputs")

files = [
    ("k1_5l", "batch_longchat_ph_k1_c256_5l.json"),
    ("k2_5l", "batch_longchat_ph_k2_c256_5l.json"),
    ("k3_5l", "batch_longchat_ph_k3_c256_5l.json"),
    ("k4_5l", "batch_longchat_ph_k4_c256_5l.json"),
    ("k2_1l", "batch_longchat_ph_k2_c256_1l.json"),
    ("c128", "batch_longchat_ph_k2_c128_5l.json"),
    ("c512", "batch_longchat_ph_k2_c512_5l.json"),
    ("mean", "batch_longchat_mean_k2_c256_5l.json"),
]

# 1. 每个配置的失败列表
print("=" * 70)
print("1. 所有配置的失败样本")
print("=" * 70)
all_data = {}
for label, fname in files:
    with open(os.path.join(OUT, fname)) as f:
        data = json.load(f)
    all_data[label] = data
    fails = [d["doc_id"] for d in data["details"] if d["selected_accuracy"] == 0]
    acc = data["summary"].get("sel_acc", data["summary"].get("avg_sel_acc", ""))
    print(f"  {label:6s}  acc={str(acc):8s}  n_fails={len(fails):2d}  {fails}")

# 2. 困难样本分析
print()
print("=" * 70)
print("2. 高难度样本 (3+ 配置失败)")
print("=" * 70)
fail_count = {}
for label, data in all_data.items():
    for d in data["details"]:
        if d["selected_accuracy"] == 0:
            did = d["doc_id"]
            fail_count[did] = fail_count.get(did, 0) + 1

hard_ids = sorted([(did, cnt) for did, cnt in fail_count.items() if cnt >= 3], key=lambda x: -x[1])
for did, cnt in hard_ids:
    print(f"  doc_id={did}: {cnt}/{len(files)} 配置失败")

# 3. k2_5l vs k2_1l 逐样本对比
print()
print("=" * 70)
print("3. 5层 vs 1层 逐样本差异 (top_k=2, c256)")
print("=" * 70)
data_5l = all_data["k2_5l"]
data_1l = all_data["k2_1l"]

both_fail = 0
both_pass = 0
only_5l_fail = []
only_1l_fail = []
for d5, d1 in zip(data_5l["details"], data_1l["details"]):
    s5 = d5["selected_accuracy"]
    s1 = d1["selected_accuracy"]
    if s5 == 0 and s1 == 0:
        both_fail += 1
    elif s5 == 1 and s1 == 1:
        both_pass += 1
    elif s5 == 0 and s1 == 1:
        only_5l_fail.append(d5["doc_id"])
    elif s5 == 1 and s1 == 0:
        only_1l_fail.append(d5["doc_id"])

print(f"  both pass:  {both_pass}")
print(f"  both fail:  {both_fail}")
print(f"  only 5l fails: {only_5l_fail} (5层多杀的)")
print(f"  only 1l fails: {only_1l_fail} (1层多杀的)")

# 4. top_k 收益递减分析
print()
print("=" * 70)
print("4. top_k 递增收益分析（仅看 k=1 失败的 17 个样本的恢复情况）")
print("=" * 70)
k1_fails = set(d["doc_id"] for d in all_data["k1_5l"]["details"] if d["selected_accuracy"] == 0)
print(f"  k=1 失败: {sorted(k1_fails)}")

for label in ["k2_5l", "k3_5l", "k4_5l"]:
    data = all_data[label]
    still_fail = k1_fails & set(d["doc_id"] for d in data["details"] if d["selected_accuracy"] == 0)
    recovered = k1_fails - still_fail
    print(f"  {label}: 恢复了 {sorted(recovered)}, 仍然失败 {sorted(still_fail)}")
