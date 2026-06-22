"""
================================================================================
 topic_boundary.py — 话题边界检测

 做什么：
   从 LongChat 的 prompt 文本中，自动检测 15 个 topic 各自占据的 token 范围。
   这些 token 范围后续可以用来 split_kv() 把完整 KV 按 topic 切开。

 输入：
   - prompt 文本（长对话记录）
   - HuggingFace tokenizer

 输出：
   - 15 个 (token_start, token_end, topic_name) 的列表

 原理：
   LongChat prompt 有固定格式：
     Topic 开始: "USER: I would like to discuss the topic of <TOPIC NAME>."
     Topic 结束: "USER: Great, this is the end of our discussion on the topic <TOPIC NAME>..."

   用正则找每个 topic 的字符起止位置 → tokenizer 把字符偏移转成 token 偏移。

 使用示例：
   from transformers import AutoTokenizer
   from topic_boundary import find_topic_token_ranges, load_longchat_sample

   data = load_longchat_sample("data/longchat.jsonl", doc_id=0)
   tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
   topics = find_topic_token_ranges(data['prompt'], tokenizer)

   for t in topics:
       print(f"Topic '{t['name']}': tokens [{t['token_start']}, {t['token_end']})")
================================================================================
"""

import re
import json

ORDINAL_WORDS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
]


def add_topic_ordinal_markers(prompt_text: str) -> str:
    """
    给 prompt 中每个 topic 的引言插入 "This is the Nth topic we discuss."

    把 "USER: I would like to discuss the topic of X."
    变成 "USER: I would like to discuss the topic of X. This is the Nth topic we discuss."

    跳过第一个匹配（系统提示里的 <TOPIC> 模板占位符）。
    从后往前替换，不破坏前面的字符偏移。
    """
    pattern = r'USER: I would like to discuss the topic of (.+?)\.'
    matches = list(re.finditer(pattern, prompt_text))
    real_matches = matches[1:]  # 跳过模板占位符

    result = prompt_text
    for i in range(len(real_matches) - 1, -1, -1):
        match = real_matches[i]
        insert_pos = match.end()
        marker = f" This is the {ORDINAL_WORDS[i]} topic we discuss."
        result = result[:insert_pos] + marker + result[insert_pos:]

    return result


def load_longchat_sample(filepath, doc_id=0):
    """
    读取 LongChat 数据集的一条样本。

    参数：
        filepath: longchat.jsonl 路径
        doc_id: 第几条样本（0-based）

    返回：
        dict，包含 prompt, label, test_id
    """
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i == doc_id:
                return json.loads(line.strip())
    raise IndexError(f"doc_id={doc_id} 超出文件行数范围")


def find_topic_char_boundaries(prompt_text):
    """
    在 prompt 文本中，用正则找每个 topic 的字符起止位置。

    LongChat prompt 的格式：
      Topic N 开始处:
        "USER: I would like to discuss the topic of <name>."

      Topic N 结束处（也是 Topic N+1 开始的前一句）:
        "USER: Great, this is the end of our discussion on the topic <name>..."

    参数：
        prompt_text: 完整的 prompt 字符串

    返回：
        list[dict]: 按顺序排列的 15 个 topic，每个包含:
            {
                'name':       话题名称（字符串，如 "the role of art in society"）,
                'char_start': prompt 中该 topic 开始的字符位置,
                'char_end':   prompt 中下一个 topic 开始的字符位置（最后一个 topic 为结束标记的末尾）,
            }
    """

    # 找所有 "I would like to discuss the topic of X."
    # [0] 是模板里的 "<TOPIC>" 占位符，跳过它取 [1:] → 15 个真正的 topic
    start_pattern = r'I would like to discuss the topic of (.+?)\.'
    start_matches = list(re.finditer(start_pattern, prompt_text))
    real_starts = start_matches[1:]  # 跳过模板占位符

    # 找所有 "end of our discussion on the topic X"
    end_pattern = r'this is the end of our discussion on the topic[^.]*\.'
    end_matches = list(re.finditer(end_pattern, prompt_text))

    if len(real_starts) != 15:
        print(f"⚠ 警告: 期望 15 个 topic，实际匹配到 {len(real_starts)} 个")

    topics = []
    for i in range(len(real_starts)):
        name = real_starts[i].group(1)
        char_start = real_starts[i].start()

        # 结束位置 = 下一个 topic 的开始（或最后一个结束标记的末尾）
        if i < len(real_starts) - 1:
            char_end = real_starts[i + 1].start()
        else:
            # 最后一个 topic：用结束标记
            char_end = end_matches[-1].end() if end_matches else len(prompt_text)

        topics.append({
            'name': name,
            'char_start': char_start,
            'char_end': char_end,
        })

    return topics


def find_topic_token_ranges(prompt_text, tokenizer):
    """
    找每个 topic 对应的 token 范围。

    原理：
      1. 用 tokenizer(prompt, return_offsets_mapping=True) 拿到每个 token
         对应的字符偏移量 (char_start, char_end)
      2. 用 find_topic_char_boundaries 拿到每个 topic 的字符边界
      3. 把字符边界映射到 token 边界

    参数：
        prompt_text: 完整的 prompt 字符串
        tokenizer:   HuggingFace tokenizer

    返回：
        list[dict]: 15 个 topic，每个包含:
            {
                'name':        话题名称,
                'token_start': topic 开始的 token 索引,
                'token_end':   topic 结束的 token 索引（不含）,
                'n_tokens':    该 topic 占用的 token 数量,
                'char_start':  字符起始位置,
                'char_end':    字符结束位置,
            }
    """

    # 拿到每个 token 的字符偏移
    encoded = tokenizer(prompt_text, return_offsets_mapping=True)
    offsets = encoded['offset_mapping']       # list[tuple(int, int)]
    input_ids = encoded['input_ids']

    def char_to_token(char_pos):
        """把字符位置映射到最近的 token 索引"""
        for tok_idx, (c_start, c_end) in enumerate(offsets):
            if c_start <= char_pos < c_end:
                return tok_idx
        # 边界情况：字符在最后一个 token 之后
        return len(offsets) - 1

    # 拿字符边界
    char_topics = find_topic_char_boundaries(prompt_text)

    token_topics = []
    for t in char_topics:
        tok_start = char_to_token(t['char_start'])
        tok_end = char_to_token(t['char_end'])

        token_topics.append({
            'name':        t['name'],
            'token_start': tok_start,
            'token_end':   tok_end,
            'n_tokens':    tok_end - tok_start,
            'char_start':  t['char_start'],
            'char_end':    t['char_end'],
        })

    return token_topics


def get_system_prompt_range(prompt_text, tokenizer):
    """
    返回「系统提示」部分的 token 范围。

    系统提示 = 第一个 topic 开始之前的所有内容。
    这部分是所有 topic 共享的上下文（角色设定、格式说明等），
    如果 query 没有特指某个 topic，可能需要保留。

    返回：
        (token_start, token_end): 系统提示的 token 范围。
                                  token_end 就是第一个 topic 的 token_start。
    """
    topics = find_topic_char_boundaries(prompt_text)
    encoded = tokenizer(prompt_text, return_offsets_mapping=True)
    offsets = encoded['offset_mapping']

    first_topic_char = topics[0]['char_start']

    def char_to_token(char_pos):
        for tok_idx, (c_start, c_end) in enumerate(offsets):
            if c_start <= char_pos < c_end:
                return tok_idx
        return len(offsets) - 1

    return (0, char_to_token(first_topic_char))


def get_final_question_range(prompt_text, tokenizer):
    """
    返回「结尾提问」部分的 token 范围。

    结尾提问 = 最后一个 topic 结束之后的内容，即：
      "Now the record ends. What is the first topic we discussed? ..."

    这部分包含了用户实际在问的问题，推理时必须保留。

    返回：
        (token_start, token_end): 结尾提问的 token 范围
    """
    topics = find_topic_char_boundaries(prompt_text)
    encoded = tokenizer(prompt_text, return_offsets_mapping=True)
    offsets = encoded['offset_mapping']

    last_topic_char_end = topics[-1]['char_end']

    def char_to_token(char_pos):
        for tok_idx, (c_start, c_end) in enumerate(offsets):
            if c_start <= char_pos < c_end:
                return tok_idx
        return len(offsets) - 1

    return (char_to_token(last_topic_char_end), len(offsets))


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    # 不需要 GPU，纯 CPU tokenizer 就能跑
    from transformers import AutoTokenizer

    # 读数据
    data = load_longchat_sample("data/longchat.jsonl", doc_id=0)
    prompt = data['prompt']
    labels = data['label']

    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    print("检测 topic 边界...\n")

    # 找 topic token 范围
    topics = find_topic_token_ranges(prompt, tokenizer)

    # 系统提示和结尾
    sys_start, sys_end = get_system_prompt_range(prompt, tokenizer)
    q_start, q_end = get_final_question_range(prompt, tokenizer)

    total_tokens = len(tokenizer(prompt).input_ids)

    print(f"总 token 数: {total_tokens}")
    print(f"系统提示:   tokens [{sys_start}, {sys_end})  ~{sys_end - sys_start} tokens")
    print(f"结尾提问:   tokens [{q_start}, {q_end}]  ~{q_end - q_start} tokens")
    print()

    print(f"{'#':<4} {'Topic Name':<55} {'tokens':<16} {'占比'}")
    print("-" * 90)

    for i, t in enumerate(topics):
        pct = t['n_tokens'] / total_tokens * 100
        print(f"{i:<4} {t['name']:<55} [{t['token_start']:>5}, {t['token_end']:>5}) "
              f"~{t['n_tokens']:>4} tokens  ({pct:.1f}%)")

    # 验证：label 和检测到的 topic name 是否一致
    print()
    print("=== 验证：检测到的 topic 名称 vs label ===")
    all_match = True
    for i, t in enumerate(topics):
        match = t['name'].lower() == labels[i].lower()
        if not match:
            print(f"  ❌ [{i}] 检测: '{t['name']}'  ≠  label: '{labels[i]}'")
            all_match = False
    if all_match:
        print("  ✅ 全部 15 个 topic 名称与 label 一致")
