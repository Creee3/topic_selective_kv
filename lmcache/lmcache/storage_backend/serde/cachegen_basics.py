import torch
import io
import pickle
from dataclasses import dataclass
from typing import List
from lmcache.utils import _lmcache_nvtx_annotate
import os

# GPU 一次算术编码最多处理 256 个 token。
# 原因：显存放不下（256 token × 4096 通道 × 每通道最多 256 字节 ≈ 需要大量显存）。
# 9500 token 的完整 KV 要分成 ceil(9500/256) = 38 批，每批独立编码。
CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK = 256


@dataclass
class CacheGenConfig:
    """
    分层量化配置：定义 32 层中每层 Key 和 Value 各用多少个 bins（量化精度）。

    设计原则（论文 Insight 1+2）：
      - Key 比 Value 精细（Key 对 attention 精度影响更大）
      - 浅层比深层精细（浅层靠近输入，捕捉低级特征，对精度更敏感）

    字段命名规则：
      key_first_layers  = 第一档覆盖多少层（从第 0 层开始数）
      key_first_bins    = 第一档用多少 bins（量化精度）
      key_second_layers = 第一档+第二档共覆盖多少层
      key_third_layers  = 总层数（= 32）
      value 同理，但只有两档（first 和 second），没有 third。

    Key 有三档，Value 有两档：
      Key:   层 0 ~ first-1       用 first_bins
             层 first ~ second-1  用 second_bins
             层 second ~ third-1  用 third_bins
      Value: 层 0 ~ first-1       用 first_bins
             层 first ~ 31        用 second_bins
    """
    key_first_layers: int
    key_second_layers: int
    key_third_layers: int
    key_first_bins: int
    key_second_bins: int
    key_third_bins: int
    value_first_layers: int
    value_first_bins: int
    value_second_bins: int

    def __getitem__(self, key: str) -> int:
        """ 让 config 对象可以像字典一样用中括号取值。
            例如 config["key_first_layers"] 等价于 config.key_first_layers。
        """
        return getattr(self, key)

    @staticmethod
    def from_model_name(model_name: str) -> "CacheGenConfig":
        """
        根据模型名 + 环境变量 QUANT_LEVEL 返回对应的量化配置。

        环境变量 QUANT_LEVEL（在 run_cachegen.py 第 95 行设置）：
          Level 1: 最小体积，压缩最猛（key: 16/16/12 bins, value: 16/12 bins）
          Level 2: 平衡档（key: 32/16/16 bins, value: 32/16 bins）— 你的实验用这个
          Level 3: 最保真（全部 32 bins）

        模型分为两个家族：
          family_7b  = 32 层 Transformer（Mistral-7B / LongChat-7B）
          family_70b = 80 层 Transformer（LongAlpaca-70B）
        """
        family_7b = ["mistralai/Mistral-7B-Instruct-v0.2",
                     "mistral-community/Mistral-7B-v0.2",
                      "lmsys/longchat-7b-16k"]
        family_70b = ["Yukang/LongAlpaca-70B-16k"]

        if "Mistral-7B" in model_name:
            # ================================================================
            # Level 1 — 最猛压缩（体积最小，损失最大）
            # Key:   前10层 16 bins, 中10层 16 bins, 后12层 12 bins
            # Value: 前2层 16 bins, 后30层 12 bins
            # 预计大小：~130-140 MB（比 Level 2 再小 ~20%）
            # ================================================================
            if os.environ["QUANT_LEVEL"] == "1":
                return CacheGenConfig(
                    key_first_layers=10,
                    key_second_layers=20,
                    key_third_layers=32,
                    key_first_bins=16,
                    key_second_bins=16,
                    key_third_bins=12,
                    value_first_layers=2,
                    value_first_bins=16,
                    value_second_bins=12
                )

            # ================================================================
            # Level 2 — 平衡档（代码默认，你的实验用这个）
            # Key:   前10层 32 bins (5-bit), 中10层 16 bins (4-bit), 后12层 16 bins
            # Value: 前2层 32 bins, 后30层 16 bins
            # 预计大小：~170 MB，压缩比 vs 8-bit baseline ≈ 3.4x
            # ================================================================
            if os.environ["QUANT_LEVEL"] == "2":
                return CacheGenConfig(
                    key_first_layers=10,
                    key_second_layers=20,
                    key_third_layers=32,
                    key_first_bins=32,
                    key_second_bins=16,
                    key_third_bins=16,
                    value_first_layers=2,
                    value_first_bins=32,
                    value_second_bins=16
                )

            # ================================================================
            # Level 3 — 最保真（体积最大，损失最小）
            # Key:   全部 32 bins (5-bit)
            # Value: 全部 32 bins
            # 预计大小：~210 MB
            # ================================================================
            if os.environ["QUANT_LEVEL"] == "3":
                return CacheGenConfig(
                    key_first_layers=10,
                    key_second_layers=20,
                    key_third_layers=32,
                    key_first_bins=32,
                    key_second_bins=32,
                    key_third_bins=32,
                    value_first_layers=2,
                    value_first_bins=32,
                    value_second_bins=32
                )

        # 非 Mistral-7B 的 7B 家族模型（如 longchat-7b），默认用 Level 2 配置
        elif model_name in family_7b:
            return CacheGenConfig(
                key_first_layers=10,
                key_second_layers=20,
                key_third_layers=32,
                key_first_bins=32,
                key_second_bins=16,
                key_third_bins=16,
                value_first_layers=2,
                value_first_bins=32,
                value_second_bins=16
            )

        # 70B 模型：80 层，层数分配比例不同（前 20 层精细，后 40 层粗糙）
        elif model_name in family_70b:
            return CacheGenConfig(
                key_first_layers=20,      # 前 20 层 → 32 bins
                key_second_layers=40,     # 第 20-39 层 → 32 bins
                key_third_layers=80,      # 第 40-79 层 → 16 bins；共 80 层
                key_first_bins=32,
                key_second_bins=32,
                key_third_bins=16,
                value_first_layers=20,    # Value 前 20 层 → 32 bins
                value_first_bins=32,
                value_second_bins=16      # 其余 60 层 → 16 bins
            )
        else:
            raise ValueError(f"Model {model_name} is not supported")


@dataclass
class CacheGenGPUBytestream:
    """
    一批 token（≤256）编码后的压缩结果。

    字段：
      bytestream: 各通道压缩后的有效字节拼接成的一维 tensor
                  编码后 64 层 × 4096 通道，每通道长度不同（变长编码），
                  有效字节紧排列成一个一维数组（padding 已去除）
      bytestream_lengths: 各通道实际用了多少字节，形状 (64, 4096)
                          解码时用来把变长字节填回固定大小的 buffer
      ntokens: 这一批编码了多少个 token（通常是 256，最后一批可能不足 256）
    """
    bytestream: torch.Tensor
    bytestream_lengths: torch.Tensor
    ntokens: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@dataclass
class CacheGenGPUEncoderOutput:
    """
    编码器完整输出——包含解码所需的全部信息。

    字段：
      data_chunks: 压缩块列表，共 ceil(ntokens/256) 个 CacheGenGPUBytestream
                   每个 chunk 编码 256 个 token
      cdf: CDF 概率表，形状 (64, 4096, 33)
           64 = 32 层 Key + 32 层 Value
           4096 = 通道数（32 头 × 128 维）
           33 = bins_max + 1 = 32 + 1
           对应论文 §5.1.3 Insight 3：按 layer×channel 建概率分布
      max_tensors_key: Key 反量化所需的缩放因子，形状 (32, ntokens, 1)
                       量化时每个 token 独立缩放，解码反量化必须用到
      max_tensors_value: Value 反量化所需的缩放因子，形状同上
      num_heads: 注意力头数（Mistral-7B = 32），用于解码后 reshape
      head_size: 每头维度（Mistral-7B = 128），用于解码后 reshape

    CDF 和 data_chunks 打包在一起传输：解码端拿到的 pickle 文件里就自带 CDF，
    不需要额外传一份"全局概率表"。每条数据自包含，独立可解。
    """
    data_chunks: List[CacheGenGPUBytestream]
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @_lmcache_nvtx_annotate
    def to_bytes(self) -> bytes:
        """
        将整个编码输出（data_chunks + CDF + max_tensors + 元信息）
        pickle 序列化成一串 bytes。约 170 MB（Level 2 时）。

        流程：io.BytesIO() 创建一个内存中的"假文件"，
        pickle.dump 往里写，f.getvalue() 取出全部字节。
        """
        with io.BytesIO() as f:
            pickle.dump(self, f)
            return f.getvalue()

    @staticmethod
    @_lmcache_nvtx_annotate
    def from_bytes(bs: bytes) -> "CacheGenGPUEncoderOutput":
        """
        反序列化：把 to_bytes 产出的 bytes 还原成 Python 对象。
        解码端调用此方法拿回 cdf + data_chunks + max_tensors。
        """
        with io.BytesIO(bs) as f:
            return pickle.load(f)


# ============================================================================
# 旧版类 CacheGenEncoderOutput（第 98-122 行）已废弃，保留仅为兼容性。
# 其功能已被 CacheGenGPUEncoderOutput 取代。
# 区别：旧版整个 KV 一次性编码成一个大的 bytestream，不分 chunk。
#       新版切成 256-token 的 chunk，支持分块传输 + 自适应压缩级别。
# ============================================================================
@dataclass
class CacheGenEncoderOutput:
    bytestream: bytes
    start_indices: torch.Tensor
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    def to_bytes(self) -> bytes:
        with io.BytesIO() as f:
            pickle.dump(self, f)
            return f.getvalue()

    @staticmethod
    def from_bytes(bs: bytes) -> "CacheGenEncoderOutput":
        with io.BytesIO(bs) as f:
            return pickle.load(f)
