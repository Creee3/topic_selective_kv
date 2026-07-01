import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import argparse
import json
import numpy as np
from collections import Counter
import re 
import string 
import pickle
import os
def f1_score(prediction, ground_truth, **kwargs):    #纯数学
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def qa_f1_score(prediction, ground_truth, **kwargs):    #包装纸
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):#去除a，an，the
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):#合并多余空格
        return " ".join(text.split())

    def remove_punc(text):#去标点
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):#全小写
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


dataset2metric = {
    "nqa": qa_f1_score,
    "tqa": qa_f1_score,
}

MAX_API_RETRY = 5
REQ_TIME_GAP = 2
DATASET_TO_PATH = {
    "longchat": "test_data/longchat.jsonl",
    "tqa": "test_data/tqa.jsonl",
    "nqa": "test_data/nqa.jsonl"
}

def get_eval(user_prompt):
    import openai

    openai.api_base = "https://api.deepseek.com"
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=500,
            )
            content = response["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(e)
            time.sleep(5)
    print(f"Failed after {MAX_API_RETRY} retries.")
    return "error"
def scorer_e(dataset, predictions, answers, all_classes):
    scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "tqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        scores += [score]
    
    return scores
def chatgpt_auto_eval(gt_result, cachegen_result):
    print("--------------- Start auto-evaluation, you should verify it does this correctly --------------")
    correct = 0
    user_prompt = f"I am testing whether a LLM model can correctly retreieve the first topic, and would like you to help me judge whether the mode ls correct. Please give me 1 for correct and 0 for incorrect. Only give me a single number. Ignore mistakes if the model is paraphasing or using synonyms. Ignore any simple mistakes such as capitalization and punctuation. The ground truth is {gt_result}, the model prediction is {cachegen_result}"

    content = get_eval(user_prompt)

    _correct = content == "1"
    correct += _correct

    output_string = "correct" if _correct else "wrong"

    print(f"Label: {gt_result}, Predict: {cachegen_result} - auto-eval goes with {output_string}")

    # To avoid rate limit by OPENAI
    time.sleep(REQ_TIME_GAP)
    return correct

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)
def calculate_acc(dataset_name, prediction, label):
    if dataset_name == "longchat":
        return chatgpt_auto_eval(label[0], prediction)
    elif dataset_name == "nqa":
        scores = scorer_e(dataset_name, [prediction], [label['answers']], [label['all_classes']])
        return scores[0]
    elif dataset_name == "tqa":
        scores = scorer_e(dataset_name, [prediction], [label['answers']], [label['all_classes']])
        return scores[0]
    
# ============================================================
# 加载模型 + tokenizer
# ============================================================
# 分两条路：
#   70B 模型（LongAlpaca-70B）→ HuggingFace 原生加载，4 张 GPU 分配
#   7B 模型（Mistral-7B/LongChat-7B）→ FastChat 的 load_model，自动分配
#
# load_8bit=True 表示把模型权重量化到 8-bit：
#   FP16 模型权重: 70亿参数 × 2字节 = 14GB
#   8-bit 模型权重: 70亿参数 × 1字节 = 7GB  → 省一半显存
#   注意：这是模型权重的量化，和 KV Cache 量化是两回事！
# ============================================================
def _dtype_from_name(dtype_name):
    if dtype_name in (None, "auto"):
        return "auto"
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype: {dtype_name}")


def define_model_and_tokenizer(
    model_id,
    num_gpus=1,
    max_gpu_memory=48,
    load_8bit=True,
    model_loader="fastchat",
    hf_quantization="none",
    hf_dtype="bf16",
    hf_attn_impl="auto",
    hf_device_map="auto",
):
    """ Define the model and tokenizer
    """
    if model_loader == "hf":
        dtype = _dtype_from_name(hf_dtype)
        load_kwargs = {
            "device_map": hf_device_map,
            "max_memory": {i: f"{max_gpu_memory}GiB" for i in range(num_gpus)},
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if dtype != "auto":
            load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = "auto"
        if hf_attn_impl != "auto":
            load_kwargs["attn_implementation"] = hf_attn_impl
        if hf_quantization != "none":
            from transformers import BitsAndBytesConfig

            compute_dtype = dtype if dtype != "auto" else torch.bfloat16
            if hf_quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            elif hf_quantization == "8bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"unknown hf_quantization: {hf_quantization}")
            load_kwargs.pop("torch_dtype", None)

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer

    # ============================================================
    # 分支1: 70B 模型（~140GB FP16 → ~70GB 8-bit）
    # 需要 4 张 A40（每张 48GB）才装得下
    # ============================================================
    if model_id == "Yukang/LongAlpaca-70B-16k":
        # device_map='auto': 自动把 80 层分配到 4 张 GPU 上
        # max_memory: 每张卡最多用 45GiB（留 3GB 给 KV Cache 等临时数据）
        from_pretrained_kwargs = {
                                'device_map': 'auto',
                                'max_memory': {0: '45GiB',
                                               1: '45GiB',
                                               2: '45GiB',
                                               3: '45GiB'},
                                'revision': 'main'}
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,    # 逐层加载到 GPU，不先把整个模型加载到 CPU 内存
                trust_remote_code=True,    # 允许运行模型仓库里的自定义 Python 代码（如 LongAlpaca 的特殊层）
                load_in_8bit=load_8bit,    # 模型权重 FP16 → int8，140GB → 70GB
                **from_pretrained_kwargs,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    # ============================================================
    # 分支2: 7B 模型（~14GB FP16 → ~7GB 8-bit）
    # 你用的 Mistral-7B 走这里！一张 A40 就够了
    # FastChat 的 load_model 内部也是调 from_pretrained，但帮你自动算了显存分配
    # ============================================================
    else:
        from fastchat.model import load_model

        model, tokenizer = load_model(
                model_id,
                device="cuda",                          # 模型放 GPU 上
                num_gpus=num_gpus,                      # GPU 数量（7B 模型 1 张就够了）
                max_gpu_memory=f"{max_gpu_memory}GiB",  # 单卡显存上限（A40 = 48GB）
                load_8bit=load_8bit,                    # 8-bit 量化 → 14GB → 7GB
                cpu_offloading=False,                   # 不把层卸载到 CPU（一张卡装得下）
                debug=False,                            # 不打印调试信息
            )


    return model, tokenizer


# ============================================================
# 5D tensor → 嵌套 tuple（to_blob 的逆操作）
# 解压后的 5D tensor (32, 2, 32, 9500, 128) 必须转回嵌套 tuple，
# 因为 model.generate(past_key_values=...) 只认这个格式
# unsqueeze(0) 加 batch 维度，.to(cuda) 搬到对应 GPU
# ============================================================
def tensor_to_tuple(kv, layer_to_device_id):
    """ Convert a tensor to a list of tuples
    Input tensor's shape should be (num_layers, 2, num_heads, seq_len, heads_dim)
    """
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0].unsqueeze(0).to(f"cuda:{layer_to_device_id[i]}"), 
                       kv[i][1].unsqueeze(0).to(f"cuda:{layer_to_device_id[i]}")))
    return tuple(new_kv)

# ============================================================
# 单层量化: FP16 → int8（均匀量化）
# ============================================================
# 量化公式: xq = round(原始值 * (MAX / max1))
#   其中 MAX = bins // 2 - 1（有符号，一半给正一半给负）
#   例如 bins=256 → MAX=127
#
# 每个 token 的每一行独立量化，max1 是这行的最大绝对值
# 这样每行有自己的缩放因子，精度比全局统一缩放更高
#
# 具体例子:
#   原始行: [0.0032, -0.0015, 0.0001, ...]
#   max1 = 0.005（这行的最大绝对值）
#   缩放: 0.0032 * (127 / 0.005) = 81.28 → round → 81
#        -0.0015 * 25400 = -38.1 → round → -38
#   结果: [81, -38, 3, ...]  (int8)
# ============================================================
def torch_quant(bins: int, qA: torch.Tensor):
    """
    Quantize a float tensor to fixed number of bins

    Input:
        bins: number of bins
        qA: the input tensor

    Returns:
        xq: the quantized tensor, in float32
        max1: the maximum value of the tensor
    """
    MAX = bins // 2 - 1           # bins=256 → 127. 有符号范围: [-127, 127]
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)  # 每行的最大绝对值 [nrows, 1]
    xq = torch.round(qA * (C / max1)).to(torch.int8)        # ★ 量化公式: 缩放 + 四舍五入 + 转 int8

    x = (xq / C * max1).to(torch.float16)  # 验证用: 立即反量化回去，这行实际没用（x 没被返回）

    return xq, max1
# ============================================================
# 单层反量化: int8 → FP16（量化的逆操作）
# ============================================================
# 反量化公式: x = xq / MAX * max1
#   其中 MAX = bins // 2 - 1（和量化时一样）
#
# 量化时: xq = round(x * (127 / max1))
# 反量化: x ≈ xq / 127 * max1
#
# 例如: xq=81, max1=0.005
#       81 / 127 * 0.005 = 0.6378 * 0.005 = 0.003189
#       原始值 = 0.003200
#       误差 = 0.000011  ← 量化误差，很小但存在
# ============================================================
def torch_dequant(bins: int, xq: torch.Tensor, max1: torch.Tensor):
    """
    Dequantize a quantized tensor

    Input:
        bins: number of bins
        xq: the quantized tensor
        max1: the maximum value of the tensor

    Returns:
        x: the dequantized tensor
    """
    MAX = bins // 2 - 1    # 和量化时一样的 MAX
    C = MAX
    x = (xq / C * max1).to(torch.float16)   # ★ 反量化公式: 除以缩放因子，乘回最大值
    return x

# ============================================================
# 整个 KV Cache 批量量化（所有 32 层）
# ============================================================
# 和 CacheGen 的量化不同:
#   - Baseline: 所有层统一 bins=256（8-bit），简单均匀量化
#   - CacheGen: 每层不同 bins（浅层 32，深层 16）+ 算术编码
#
# 流程:
#   遍历 32 层:
#     1. reshape: (头数, 词数, 维度) → (词数, 头数*维度) = (9500, 4096)
#        ★ 为什么要 reshape？torch_quant 按行量化，
#           把 32 个头 × 128 维 = 4096 作为一行，对每个 token 独立量化
#     2. torch_quant: FP16 → int8，返回量化值和 max 值
#     3. reshape 回去: (9500, 4096) → (头数, 词数, 维度)
#     4. 收集每层的 max_tensors（反量化时必需）
# ============================================================
def default_quantization(kv, bins, layer_to_device_id):
    """ Quantize the key value tensors into tuple of key and value tensors
    """
    channels = kv.shape[-1] * kv.shape[-3]   # 128维 × 32头 = 4096（每个 token 的特征拼接成一长行）
    max_tensors = None
    for i in range(len(kv)):      # 遍历 32 层
        key = kv[i][0]            # (32头, 9500词, 128维)
        value = kv[i][1]          # (32头, 9500词, 128维)
        # --- reshape: (头数, 词数, 维数) → (词数, 头数×维数) ---
        #     permute((1, 0, 2)): 头数和词数交换 → (9500, 32, 128)
        #     reshape: 合并后两维 → (9500, 4096)
        key = key.permute((1, 0, 2)).reshape(kv.shape[-2], channels)
        value = value.permute((1, 0, 2)).reshape(value.shape[-2], channels)
        # --- 量化: FP16 → int8 ---
        key, maxk = torch_quant(bins, key)
        value, maxv = torch_quant(bins, value)
        # --- reshape 回去: (9500, 4096) → (头数, 词数, 维数) ---
        quant_key = key.reshape(kv[i][0].shape[-2], kv[i][0].shape[-3], kv[i][0].shape[-1]).permute((1, 0, 2))
        quant_value = value.reshape(kv[i][1].shape[-2], kv[i][1].shape[-3], kv[i][1].shape[-1]).permute((1, 0, 2))
        # --- 替换为量化后的 K/V ---
        kv[i][0] = quant_key
        kv[i][1] = quant_value
        # --- 保存 max 值: (2, 9500, 1) 堆叠 → (32层, 2种, 9500, 1) ---
        concated_max = torch.cat((maxk.unsqueeze(0), maxv.unsqueeze(0)), dim=0)
        if max_tensors is None:
            max_tensors = concated_max.unsqueeze(0)
        else:
            max_tensors = torch.cat((max_tensors, concated_max.unsqueeze(0)), dim=0)
    return kv.to(torch.int8), max_tensors   # 5D tensor (32,2,32,9500,128) int8 + max表

# ============================================================
# 整个 KV Cache 批量反量化（default_quantization 的逆操作）
# ============================================================
# 流程（和量化完全对称）:
#   1. int8 → float16 容器
#   2. 遍历 32 层:
#       reshape (头,词,维) → (词, 头×维)
#       torch_dequant: int8 → FP16（用之前保存的 max_tensors）
#       reshape 回去
#       kv[i] 替换为反量化后的 FP16 值
#   3. tensor_to_tuple: 5D tensor → 模型能读的嵌套 tuple
# ============================================================
# ★ 注意: 反量化后的值和原始 FP16 不完全一样！
#    量化时: xq = round(x * 127 / max1)     ← 四舍五入丢失了小数点
#    反量化: x' = xq / 127 * max1             ← 恢复的值有微小误差
#    但 8-bit (256 bins) 误差足够小，对推理质量影响不大
# ============================================================
def dequantize_kv(kv, max_tensors, args, layer_to_device_id):
    channels = kv.shape[-1] * kv.shape[-3]   # 4096
    kv = kv.to(torch.float16)                 # int8 → float16 容器（准备装反量化结果）
    for i in range(len(kv)):
        key = kv[i][0]
        value = kv[i][1]
        # --- 同样的 reshape: (头数, 词数, 维数) → (词数, 头数×维数) ---
        key = key.permute((1, 0, 2)).reshape(kv.shape[-2], channels)
        value = value.permute((1, 0, 2)).reshape(value.shape[-2], channels)
        # --- 反量化: int8 → FP16（用之前保存的 max 值恢复缩放） ---
        dequant_k = torch_dequant(args.bins, key, max_tensors[i][0])   # Key 反量化
        dequant_v = torch_dequant(args.bins, value, max_tensors[i][1]) # Value 反量化
        # --- reshape 回去: (9500, 4096) → (头数, 词数, 维数) ---
        dequant_key = dequant_k.reshape(kv[i][0].shape[-2], kv[i][0].shape[-3], kv[i][0].shape[-1]).permute((1, 0, 2))
        dequant_value = dequant_v.reshape(kv[i][1].shape[-2], kv[i][1].shape[-3], kv[i][1].shape[-1]).permute((1, 0, 2))
        # --- 替换为反量化后的 K/V ---
        kv[i][0] = dequant_key
        kv[i][1] = dequant_value
    return tensor_to_tuple(kv, layer_to_device_id)   # 5D tensor → 嵌套 tuple，模型直接能用

#打破字符串的外壳，露出里面真正的数据结构
def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases


def bw_generator(num_chunks):
    import numpy as np
    import random
    min = 0.1
    max = 10
    bw = np.zeros(num_chunks)
    for i in range(num_chunks):
        bw[i] = random.uniform(min, max)
    return bw

def profile(model, args):
    st = time.monotonic()
    input_ids = torch.randint(0, 32000, (1, args.chunk_size)).cuda()
    
    model.generate(input_ids,  do_sample=False,  max_new_tokens=1)
    torch.cuda.synchronize()
    return time.monotonic() - st


def bw_generator(num_chunks):
    import numpy as np
    import random
    min = 0.1
    max = 10
    bw = np.zeros(num_chunks)
    for i in range(num_chunks):
        bw[i] = random.uniform(min, max)
    return bw

# ================================================================
# config_selection: 自适应选压缩级别
# ================================================================
# 对每个 chunk，根据当前带宽从 Q=3→2→1 尝试，选第一个能塞进时间预算的。
# 三个全挂 → fallback (config=0, 惩罚 0.2s)。
#
# 输入:
#   all_bws:     带宽轨迹, 如 [3.2, 0.8, 5.1, ...]  (每个 chunk 一个带宽值)
#   chunk_delay:  fallback 惩罚 = 0.2s
#   args:         含 chunk_size, slo, save_dir
#   length:       token 总数 (~9500)
#   doc_id:       当前 doc 编号
#
# 输出:
#   ttft:     总耗时 (秒)
#   configs:  每个 chunk 选了什么级别, 如 [2, 0, 3, 2, 1, ...]
#             2=Level2, 0=fallback(网络太差放弃传输), 3=Level3
# ================================================================
def config_selection(all_bws, chunk_delay, args, length, doc_id):
    # 多少个 chunk: 9500/1000 = 10 个
    num_chunks = round(length / args.chunk_size)
    # 每个 chunk 的预算: SLO / chunk 数。SLO=0.5s → 50ms
    time_budget_per_chunk = args.slo / num_chunks

    chunk_id = 0
    ttft = 0       # 累计总传输时间
    configs = []   # 每个 chunk 最终选了哪个级别

    for chunk_start in range(0, length, args.chunk_size):
        bw = all_bws[chunk_id]   # 当前 chunk 的模拟带宽 (Gbps)
        found_cache = False

        # 从高质量到低质量尝试: Q=3 → Q=2 → Q=1
        for quant_level in np.arange(3, 0, -1):   # 3, 2, 1
            # 读离线编码好的文件: {doc_id}_{chunk_id}_{quant_level}.pkl
            bytestream = pickle.load(open(
                f"{args.save_dir}/{doc_id}_{chunk_id}_{quant_level}.pkl", "rb"
            ))
            # 传输时间 = 文件大小(bits) / 带宽(bps)
            transmission_time = len(bytestream) / 1e9 * 8 / bw

            if transmission_time < time_budget_per_chunk:
                # 塞得进预算！选这个级别
                ttft += transmission_time
                found_cache = True
                configs.append(quant_level)
                break   # 跳出 quality loop, 继续下一个 chunk

        if not found_cache:
            # 三个级别全超时 → fallback
            # 服务器直接拿原始文本重算 prefill（实验里用 split_kv 等价替代）
            # 惩罚 0.2s
            ttft += chunk_delay
            configs.append(0)    # 0 = fallback

        chunk_id += 1

    return ttft, configs
def merge_kv(left, right, free_left = False, free_right = False):
    """
    Merges two kv caches, returns a merged KV cache
    A single KVCache is a tuple_32(tuple_2(torch.Tensor[bs, channels?, num_tokens, hidden_size]))

    Input:
    - left: the left kv cache, could be None
    - right: the right kv cache

    Returns: The merged kv cache. If left is None, returns right
    """
    if left is None:
        return right
    #assert len(left) == len(right)

    def generator():
        for left_layer, right_layer in zip(left, right):
            yield (torch.cat([left_layer[0], right_layer[0]], dim = -2), torch.cat([left_layer[1], right_layer[1]], dim = -2))
            if free_left:
                del left_layer
            if free_right:
                del right_layer

    return tuple(generator())
def split_kv(kv, left: int, right: int):
    """
    Splits a kv cache into two kv caches
    A single KVCache is a tuple_32(tuple_2(torch.Tensor[bs, channels?, num_tokens, hidden_size]))

    Input:
    - kv: the kv cache to be splitted
    - split_index: the index to split the kv cache

    Returns: a tuple of two kv caches
    """
    
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0][:, left:right].unsqueeze(0), 
                       kv[i][1][:, left:right].unsqueeze(0)))
    return tuple(new_kv)

# ================================================================
# merge: 根据 configs 把多个 chunk 拼回完整 KV
# ================================================================
# configs 是 config_selection 的输出，比如 [2, 0, 3, 2, 1, ...]。
# 对每个 chunk:
#   config=0 → fallback: 从原始 KV 切一块（模拟服务器重算 prefill）
#   config≠0 → 读离线编码好的 .pkl 文件 → CacheGen 解码
# 然后 merge_kv 沿 token 维拼起来。
#
# 输入:
#   configs:  每个 chunk 选的级别, [2, 0, 3, ...]
#   orig_kv:  完整的原始 KV Cache (tuple 格式), fallback 时用
#   layer_to_device_id:  每层 KV 在哪个 GPU 上
#
# 输出:
#   完整拼好的 KV Cache (tuple 格式, 可直接注入 model.generate)
# ================================================================
def merge(configs, args, doc_id, length, orig_kv=None, layer_to_device_id=None):
    merged_kv = None   # 从空开始, 逐个 chunk 往上拼
    chunk_id = 0

    for chunk_start in range(0, length, args.chunk_size):
        # 最后一个不完整的 chunk 跳过（数据长度刚好整除时不会触发）
        if chunk_start + args.chunk_size > length:
            break

        # ----------------------------------------------------------------
        # 情况 1: config=0 → fallback
        # ----------------------------------------------------------------
        # 网络太差, 连 Q=1 都传不完 → 放弃传输！
        # 服务器直接拿原始文本重做 prefill，结果等价于从完整 KV 切一块。
        # 实验里用 split_kv 代替真正的重算（速度快但结果一样）。
        # chunk_delay=0.2s 的惩罚已经加在 config_selection 的 ttft 里了。
        # ----------------------------------------------------------------
        if configs[chunk_id] == 0:
            loaded_kv = split_kv(
                orig_kv,
                chunk_start,                          # 从哪个 token 开始
                chunk_start + args.chunk_size         # 到哪个 token 结束
            )
            # loaded_kv: 嵌套 tuple, 每层 K/V 形状 (1, 32, chunk_size, 128)

        # ----------------------------------------------------------------
        # 情况 2: config≠0 → 读离线编码文件, CacheGen 解码
        # ----------------------------------------------------------------
        # 和 run_cachegen.py 阶段2 的解码完全一样:
        #   读 pickle → CacheGenDeserializer.from_bytes → tensor_to_tuple
        else:
            # 读编码好的文件: {doc_id}_{chunk_id}_{level}.pkl
            from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
            from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer

            os.environ["QUANT_LEVEL"] = str(configs[chunk_id])
            loaded_bytes = pickle.load(open(
                f"{args.save_dir}/{doc_id}_{chunk_id}_{configs[chunk_id]}.pkl",
                "rb"
            ))
            # 创建解压器
            lmcache_config = LMCacheEngineConfig.from_defaults(
                chunk_size=args.chunk_size
            )
            meta_data = LMCacheEngineMetadata(
                model_name=args.model_id, fmt="huggingface",
                world_size=1, worker_id=0
            )
            deserializer = CacheGenDeserializer(lmcache_config, meta_data)
            # GPU 解码: 算术解码 + 反量化 + 拼 5D tensor
            decoded_kv = deserializer.from_bytes(loaded_bytes)
            # 5D tensor → 嵌套 tuple (模型格式)
            loaded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)

        # ----------------------------------------------------------------
        # 把当前 chunk 拼到之前累积的 KV 上
        # ----------------------------------------------------------------
        # merge_kv 沿 token 维拼接两个 KV:
        #   第1次: None + chunk0 → chunk0
        #   第2次: chunk0 + chunk1 → chunk0+1
        #   ...
        #   第N次: chunk0+1+...+N-1 + chunkN → 完整 9500 token KV
        merged_kv = merge_kv(merged_kv, loaded_kv)
        chunk_id += 1

    return merged_kv   # 完整 KV Cache, 可直接注入 model.generate
