"""
================================================================================
 cachegen_encoder.py — CacheGen 编码器（KV Cache 压缩）
================================================================================

调用链（从 run_cachegen.py 追踪）:
  run_cachegen.py:138  bytes = cachegen_serializer.to_bytes(key_value)
       │
       ▼
  CacheGenSerializer.to_bytes() [第325行]
       │
       ├── 第341-342行: permute (HuggingFace → CacheGen 格式)
       │   (32,2,32,9500,128) → (32,2,9500,32,128)
       │
       └── encode_function() [第247行]
             ├── 第257行: _split_kv → 拆成 Key 和 Value
             ├── 第261-262行: torch_quant_vectorized → 分层量化 (FP16→int8)
             ├── 第265-267行: calculate_cdf → GPU 统计概率分布 (CUDA)
             └── 第282行: encode_ntokens → GPU 算术编码 (CUDA)

整体流程:
  原始 KV tensor (1.2GB) → permute → split → 量化 → CDF → AC编码 → 字节流 (~170MB)
================================================================================
"""

import io
import pickle
import torchac
import torchac_cuda
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, List, Any

from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig, CacheGenEncoderOutput, CacheGenGPUBytestream, CacheGenGPUEncoderOutput
import lmcache.storage_backend.serde.cachegen_basics as CGBasics
from lmcache.storage_backend.serde.serde import Serializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

# ============================================================
# 单层量化函数（old version，逐层调用）
# 把 FP16 浮点数 → 0~(bins-1) 的 int8 整数
# ============================================================
@_lmcache_nvtx_annotate
def torch_quant(bins: int, qA: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    对单层 tensor 做均匀量化

    参数:
        bins: 量化精度，如 32 bins → 值域 [0, 31] (5 bit)
        qA:   输入 tensor，shape = [ntokens, nchannels]

    返回:
        xq:   量化后的 int8 值
        max1: 每 token 的最大绝对值（解码反量化时必须用到）

    公式:
        MAX = bins // 2 - 1          (32 bins → MAX=15)
        max1 = max(|qA|, dim=-1)     每个 token 在所有 channel 上的最大绝对值
        xq = round(qA * (MAX / max1)) 缩放后 round 到 int8
    """
    MAX = bins // 2 - 1           # 32→15, 16→7
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)  # 每 token 最大绝对值
    xq = torch.round(qA * (C / max1)).to(torch.int8)        # 缩放 + round + 转 int8

    x = (xq / C * max1).to(torch.float32)                   # (未使用，仅供参考)

    return xq, max1

# ============================================================
# 批量量化函数（new version，用 GPU 批处理所有层）
# 一次处理 32 层，每层用不同的 bins
# ============================================================
@_lmcache_nvtx_annotate
def torch_quant_vectorized(bins: torch.Tensor, input_groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对所有层批量做量化

    参数:
        bins:        每层的量化精度，shape = [nlayers]  如 [32,32,...,16,16]
        input_groups: 输入 tensor，shape = [nlayers, ntokens, nchannels]

    返回:
        quantized groups: [nlayers, ntokens, nchannels] 量化后的 int8
        maxes:            [nlayers, ntokens, 1]          每 token 的最大值（反量化用）

    核心公式:
        MAX = (bins // 2 - 1)[:, None, None]   [nlayers, 1, 1]
        max1 = max(|input|, dim=-1)             [nlayers, ntokens, 1]
        xq = round(input * (MAX/max1) + MAX)    [nlayers, ntokens, nchannels]
        （+MAX 是为了把值从 [-MAX, MAX] 平移到 [0, 2*MAX]）
    """
    MAX = (bins // 2 - 1)[:, None, None]  # shape [nlayers, 1, 1]，广播到所有 token 和 channel
    max1 = torch.amax(torch.abs(input_groups), dim=-1, keepdim=True)  # shape [nlayers, ntokens, 1]
    factor = MAX / max1                   # shape [nlayers, ntokens, 1]，缩放因子
    xq = torch.round(input_groups * factor + MAX).to(torch.int8)  # 量化！+MAX 平移到非负

    return xq, max1

# ============================================================
# 把 dict 里的 max tensor 按层拼接成一个大 tensor（辅助函数）。#可忽略
# ============================================================
@_lmcache_nvtx_annotate
def concat_max(max1):
    """
    将 {layer_id: tensor} 拼接成单个 tensor [nlayers, ...]
    """
    # TODO: this function can be optimized, we don't really need this
    maxes = []
    for i in range(len(max1)):
        maxes.append(max1[i].unsqueeze(0))
    return torch.cat(maxes, dim=0)

# ============================================================
# 拆分 KV blob tensor 为独立的 Key 和 Value
# ============================================================
def _split_kv(tensor: torch.Tensor) -> torch.Tensor:
    """
    把 5D blob tensor 拆成 K 和 V 两个 tensor

    输入:
        tensor: [num_layers, 2, num_tokens, num_heads, head_size]
                如 (32, 2, 9500, 32, 128)

    返回:
        K 和 V: 各 [num_layers, num_tokens, num_channels]
                如 (32, 9500, 4096)
                其中 num_channels = num_heads * head_size = 32*128 = 4096

    步骤:
        1. reshape: 合并 heads 和 head_size → (32, 2, 9500, 4096)
        2. unbind dim=1: 沿 "2" 那个维度拆开 → 两份 (32, 9500, 4096)
    """
    num_layers, _, num_tokens, num_heads, head_size = tensor.shape
    # 合并 heads 维度: 32头×128维 = 4096 channel
    # unbind dim=1: 把 (32, 2, 9500, 4096) 沿 dim=1 拆成 [K, V]
    return torch.unbind(tensor.reshape(num_layers, 2, num_tokens, num_heads * head_size), dim=1)

# ============================================================
# CDF 归一化：浮点 CDF → int16 CDF（算术编码需要整数格式）
# ============================================================
@_lmcache_nvtx_annotate
def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    """
    将 [0,1) 的浮点 CDF 转为 [0, 2^16) 的 int16 CDF

    为什么?
      算术编码要求 CDF 严格单调递增，且必须是整数。
      float → int 转换中可能出现重复值，破坏单调性。
      解决: 乘以一个大数后，加上 arange(Lp) 保证严格递增。
    """
    PRECISION = 16
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(
      2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
      new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
      r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
      cdf.add_(r)
    return cdf

# ============================================================
# CacheGenEncoderImpl — 旧版编码器类（逐层量化 + 逐层 CDF）
# 保留未使用，新版用 encode_function 直接处理
# ============================================================
class CacheGenEncoderImpl:
    def __init__(self, **kwargs) -> None:
        """
        输入:
          fp_k / fp_v: 已拆分的 K 和 V tensors
          config:       CacheGenConfig 量化配置
        """
        self.fp_k = kwargs["fp_k"]
        self.fp_v = kwargs["fp_v"]

        self.quantized_key = {}
        self.max_tensors_key = {}
        self.quantized_value = {}
        self.max_tensors_value = {}
        self.config = kwargs["config"]

    @_lmcache_nvtx_annotate
    def quantize(self):
        """
        逐层量化 Key 和 Value
        根据每层编号决定用哪个 bins:
          key:   层<first_layers → first_bins,  层<second_layers → second_bins,  其余 → third_bins
          value: 层<first_layers → first_bins,  其余 → second_bins
        """
        # --- 量化 Key ---
        for layer in range(len(self.fp_k)):
            # 判断当前层属于哪个区间
            if layer < self.config["key_first_layers"]:
                bins = self.config["key_first_bins"]
            elif layer < self.config["key_second_layers"]:
                bins = self.config["key_second_bins"]
            else:
                bins = self.config["key_third_bins"]

            tmp = torch_quant(bins, self.fp_k[layer].float())
            self.quantized_key[layer] = tmp[0] + bins // 2 - 1  # 加偏移保证非负
            self.max_tensors_key[layer] = tmp[1]

        # --- 量化 Value ---
        for layer in range(len(self.fp_v)):
            if layer < self.config["value_first_layers"]:
                bins = self.config["value_first_bins"]
            else:
                bins = self.config["value_second_bins"]
            tmp = torch_quant(bins, self.fp_v[layer].float())
            self.quantized_value[layer] = tmp[0]+ bins // 2 - 1  # 加偏移保证非负
            self.max_tensors_value[layer] = tmp[1]

    @_lmcache_nvtx_annotate
    def compute_cdf(self, is_key):
        """
        对量化后的 tensor 统计 CDF（累积分布函数）

        步骤:
          1. 对每个 channel，统计每个量化值的出现次数 (one_hot + sum)
          2. 归一化 (除以 token 数)
          3. cumsum 得到累积分布
        """
        channels = self.fp_k[0].shape[-1]
        tokens = self.fp_k[0].shape[0]

        def process_batch(X, max_val):
            """
            统计一批 tensor 的 CDF
            input shape: [channels, tokens]
            返回: [nchannels, max_val+1] 的累积分布
            """
            nchannels, ntokens = X.shape
            # one_hot: 每个值变成 max_val+1 维的独热向量
            one_hot = torch.nn.functional.one_hot(X.long(), num_classes=max_val + 1).to(torch.float32)
            # 累加每个 channel 中每个值出现次数 → 归一化
            counts = one_hot.sum(dim=1) / ntokens
            # cumsum → CDF，roll(1) 向右平移一位（CDF 从 0 开始）
            ret = torch.cumsum(counts, dim=1).roll(1)
            ret[:, 0] = 0
            return ret

        def process_layers(X, max_val):
            """
            逐层处理，最后拼成一个大的 CDF tensor
            """
            results = []
            for x in X:
                batch_counts = process_batch(x.cuda().permute(1, 0), max_val)
                results.append(batch_counts)
            final_counts = torch.cat(results, dim=0)
            return final_counts

        if is_key:
            X = self.quantized_key.values()
        else:
            X = self.quantized_value.values()
        value_range = 32  # 最大 bins 数
        cdfs = process_layers(X, value_range)
        final_cdf = cdfs.reshape((len(self.fp_k), channels, value_range+1))

        return final_cdf

# ============================================================
# 收集压缩字节（从 output_buffer 中提取有效字节）
# ============================================================
@_lmcache_nvtx_annotate
def collect_bytes(output_buffer, output_lengths) -> torch.Tensor:
    """
    算术编码输出是变长的（每个 channel 编码后长度不同）。
    这个函数从固定大小的 buffer 中按 lengths 提取有效字节。

    输入:
        output_buffer:  [nlayers, nchannels, BUFFER_SIZE]  编码输出缓冲区
        output_lengths: [nlayers, nchannels]               每个 channel 的有效长度

    返回:
        byte_tensor: 拼接后的有效字节（去除 padding）
    """
    output_buffer_size = output_buffer.shape[-1]
    flattened_lengths = output_lengths.flatten()
    flattened_buffer = output_buffer.flatten()
    # 计算每个 channel 的起始位置
    summed_length = (output_buffer_size - flattened_lengths).cumsum(0)
    summed_length = summed_length.roll(1)
    summed_length[0] = 0
    # 根据起始位置和长度提取有效字节
    indexes = summed_length.repeat_interleave(flattened_lengths)
    indexes = indexes + torch.arange(len(indexes), device=indexes.device)
    return flattened_buffer[indexes]

# ============================================================
# GPU 算术编码入口（批次处理）
# ============================================================
@_lmcache_nvtx_annotate
def encode_ntokens(cdf_int, encode_input, output_buffer, output_lengths) -> torch.Tensor:
    """
    对一批 token 执行 GPU 算术编码

    输入:
        cdf_int:        [nlayers, nchannels, Lp]  CDF 概率表 (int16)
        encode_input:   [nlayers, ntokens, nchannels]  要编码的量化 int8 值
        output_buffer:  [nlayers, nchannels, BUFFER_SIZE]  输出缓冲区
        output_lengths: [nlayers, nchannels]  输出各 channel 的编码后长度

    返回:
        byte_tensor: 有效压缩字节（去除了 padding）

    核心: torchac_cuda.encode_fast_new() — 自定义 CUDA kernel
    """
    # 调 CUDA kernel 做算术编码
    torchac_cuda.encode_fast_new(
            cdf_int,
            encode_input,
            output_buffer,
            output_lengths,
    )
    # 从 buffer 中提取有效字节
    byte_tensor = collect_bytes(output_buffer, output_lengths)
    return byte_tensor

# ============================================================
# encode_function — 编码主函数（整个压缩流程的核心）
# ============================================================
@_lmcache_nvtx_annotate
def encode_function(
        kv: torch.Tensor,
        config: CacheGenConfig,
        key_bins: torch.Tensor,
        value_bins: torch.Tensor,
        chunk_size: int) -> CacheGenGPUEncoderOutput:
    """
    对整个 KV Cache 执行压缩

    输入:
        kv:          5D tensor (32, 2, 9500, 32, 128)  — CacheGen 格式
        config:      量化配置 (CacheGenConfig)
        key_bins:    每层 Key 的量化精度 [32] = [32,32,...,16,16]
        value_bins:  每层 Value 的量化精度 [32]
        chunk_size:  KV 的 token 数 (9500)

    流程:
        1. _split_kv        → 拆成 K 和 V 各 (32, 9500, 4096)
        2. torch_quant_vectorized → 分层量化 (FP16 → int8)
        3. calculate_cdf    → GPU 统计概率分布
        4. encode_ntokens   → GPU 算术编码 → 压缩字节流
    """
    num_heads, head_size = kv.shape[-2:]           # 32, 128
    # --- 步骤1: 拆分 K/V ---
    fp_k, fp_v = _split_kv(kv)                     # 各 (32, 9500, 4096)
    nchannels = num_heads * head_size               # = 4096
    nlayers = fp_k.shape[0] + fp_v.shape[0]         # = 64 (32K + 32V)

    # --- 步骤2: 分层量化 ---
    new_key, max_tensors_key = torch_quant_vectorized(key_bins, fp_k)
    new_value, max_tensors_value = torch_quant_vectorized(value_bins, fp_v)
    encode_input = torch.cat((new_key, new_value), dim=0).reshape(nlayers, chunk_size, nchannels)
    # encode_input shape: (64, 9500, 4096)

    # --- 步骤3: GPU 计算 CDF（概率分布） ---
    new_cdf_key = torchac_cuda.calculate_cdf(new_key, int(key_bins.max()))
    new_cdf_value = torchac_cuda.calculate_cdf(new_value, int(value_bins.max()))
    cdf_int = torch.cat([new_cdf_key, new_cdf_value])  # (64, 4096, bins+1)
    # cdf_int 是算术编码的"字典"，解码时必须用到

    # --- 步骤4: GPU 算术编码 ---
    # 准备输出缓冲区（每 256 token 一处理，因为 GPU 显存有限）
    output_buffer = torch.zeros(
            (nlayers, nchannels, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK),  # (64, 4096, 256)
            dtype=torch.uint8,
            device=encode_input.device)
    output_lengths = torch.zeros(
            (nlayers, nchannels),
            dtype=torch.int32,
            device=encode_input.device)

    # 分批处理：每批最多 256 token
    data_chunks = []
    for i in range(0, chunk_size, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
        start = i
        end = min(i + CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
        bytestream = encode_ntokens(
            cdf_int,
            encode_input[:, start:end, :],    # 切片: (64, 256, 4096)
            output_buffer,
            output_lengths
        )
        data_chunks.append(CacheGenGPUBytestream(
            bytestream = bytestream,           # 压缩字节
            bytestream_lengths = output_lengths.clone(),  # 各 channel 长度
            ntokens = end - start,             # 这批处理了多少 token
        ))

    return CacheGenGPUEncoderOutput(
            data_chunks,                       # 38 个压缩块
            cdf_int,                           # CDF 表（解码必须用！）
            max_tensors_key = max_tensors_key,     # 量化缩放因子（反量化必须用！）
            max_tensors_value = max_tensors_value, # 同上
            num_heads = num_heads,             # = 32
            head_size = head_size,             # = 128
        )

# ============================================================
# CacheGenSerializer — 编码器对外的 API（run_cachegen.py 调用的就是这个）
# ============================================================
class CacheGenSerializer(Serializer):
    """
    CacheGen 序列化器（压缩器）

    使用方式:
        serializer = CacheGenSerializer(config, metadata)
        bytes = serializer.to_bytes(kv_tensor)   # 5D tensor → 压缩字节流
    """
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        # --- 根据模型名查表，拿到分层量化配置 ---
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        # 例如 Mistral-7B, QUANT_LEVEL=2:
        #   key_first_layers=10, key_first_bins=32
        #   key_second_layers=20, key_second_bins=16
        #   key_third_layers=32, key_third_bins=16
        #   value_first_layers=2, value_first_bins=32
        #   value_second_bins=16

        self.chunk_size = config.chunk_size      # = 9500 (整个 KV 作为一个 chunk)
        self.fmt = metadata.fmt                  # = "huggingface"

        # --- 把分层配置变成"每层一个数字"的数组 ---
        self.key_bins = self.make_key_bins(self.cachegen_config)
        # 结果: [32,32,32,32,32,32,32,32,32,32,  ← 层0-9:   32 bins (5 bit)
        #        16,16,16,16,16,16,16,16,16,16,  ← 层10-19: 16 bins (4 bit)
        #        16,16,16,16,16,16,16,16,16,16,16,16] ← 层20-31: 16 bins (4 bit)
        self.value_bins = self.make_value_bins(self.cachegen_config)
        # 结果: [32,32, 16,16,16,...]  ← 层0-1: 32 bins, 层2-31: 16 bins

    # ============================================================
    # 生成 Key 的 per-layer bins 数组
    # ============================================================
    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        """
        先生成全 third_bins 的数组，再覆盖 first 和 second

        例如 QUANT_LEVEL=2:
          (1) 全填 16 → [16,16,...,16]
          (2) 前20改为16 → (没变化)
          (3) 前10改为32 → [32,32,...,32, 16,16,...,16]
                              └─ 0-9 ─┘  └─ 10-31 ─┘
        """
        ret = torch.zeros(config.key_third_layers)  # 32 个 0
        ret.fill_(config.key_third_bins)             # 全填 third_bins (如 16)
        ret[:config.key_second_layers] = config.key_second_bins  # 前 second_layers 层覆盖
        ret[:config.key_first_layers] = config.key_first_bins    # 前 first_layers 层覆盖
        return ret.cuda()

    # ============================================================
    # 生成 Value 的 per-layer bins 数组
    # ============================================================
    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        """
        同 make_key_bins，但 Value 只有两档（first 和 second）
        例如: [32,32, 16,16,16,16,16,...]
               └0-1┘ └──── 2-31 ──────┘
        """
        ret = torch.zeros(config.key_third_layers)  # 同样 32 层
        ret.fill_(config.value_second_bins)          # 全填 second_bins
        ret[:config.value_first_layers] = config.value_first_bins  # 前 first_layers 层覆盖
        return ret.cuda()

    # ============================================================
    # to_bytes — 对外的压缩接口
    # ============================================================
    @_lmcache_nvtx_annotate
    def to_bytes(
            self,
            tensor: torch.Tensor
        ) -> bytes:
        """
        把 HuggingFace 格式的 KV tensor 压缩成字节流

        输入:
            tensor: 5D tensor, shape (32, 2, 32, 9500, 128)
                    这就是 main.py 产出、torch.load 读入的那个 blob tensor

        返回:
            bytes: 压缩后的字节流 (~170MB, 约为原始的 1/7)

        流程:
            1. permute: (32,2,32,9500,128) → (32,2,9500,32,128)
               HuggingFace 格式 → CacheGen 内部格式（heads 和 tokens 换位置）
            2. encode_function: 量化 + CDF + 算术编码
            3. pickle.dump: 打包所有数据（含 CDF + max_tensors + data_chunks）→ 字节流
        """
        # HuggingFace 格式需要转置: heads 和 tokens 维度互换
        if self.fmt == "huggingface":
            tensor = tensor.permute(0, 1, 3, 2, 4)  # (L,2,H,T,D) → (L,2,T,H,D)

        ntokens = tensor.shape[2]  # = 9500
        # 调编码主函数
        output_dict = encode_function(tensor.cuda(), self.cachegen_config,
                                      self.key_bins, self.value_bins, ntokens)
        # pickle 序列化整个输出对象 → 字节流
        return output_dict.to_bytes()
