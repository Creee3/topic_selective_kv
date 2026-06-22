"""
================================================================================
 cachegen_decoder.py — CacheGen 解码器（KV Cache 解压）
================================================================================

调用链（从 run_cachegen.py 追踪）:
  run_cachegen.py:181  decoded_kv = deserializer.from_bytes(bytes)
       │
       ▼
  CacheGenDeserializer.from_bytes() [第143行]
       │
       ├── 第145行: pickle.load → 取出编码时保存的数据
       │     (data_chunks, cdf, max_tensors_key, max_tensors_value)
       │
       ├── 第151行: decode_function_gpu() → 定义在第78行
       │     ├── 第101行: 准备空容器 (64, 9500, 4096)
       │     ├── 第103-107行: 逐个 chunk 调 decode_chunk()
       │     │      └── 第71行: torchac_cuda.decode_fast_prefsum() ← CUDA kernel
       │     └── 第109-112行: 拆成 Key 和 Value
       │
       ├── 第159-160行: do_dequantize() → 定义在第24行（反量化）
       │
       └── 第164-173行: stack + reshape + permute → 5D tensor
            (32,2,32,9500,128)  跟压缩前一样！

编码和解码完全对称（互逆）:
  编码                                   解码
  ────                                   ────
  permute (换顺序)                       permute (换回来)
  _split_kv (拆成K/V)                     stack (合回去)
  torch_quant_vectorized (FP16→int8)      do_dequantize (int8→FP16)
  calculate_cdf (算概率)                   (用同样的 CDF)
  encode_fast_new (算术编码)              decode_fast_prefsum (算术解码)
================================================================================
"""

import io
import pickle
import torchac_cuda
import numpy as np
import torch
from typing import Tuple, List, Any

from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig, CacheGenEncoderOutput, CacheGenGPUBytestream, CacheGenGPUEncoderOutput
import lmcache.storage_backend.serde.cachegen_basics as CGBasics
from lmcache.storage_backend.serde.serde import Deserializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.logging import init_logger
import nvtx

logger = init_logger(__name__)

# ============================================================
# 单层反量化（old version，直接还原）
# ============================================================
@_lmcache_nvtx_annotate
def quant(bins: int, xq: torch.Tensor, max1: float):
    """
    反量化公式:  x = xq / C * max1
    其中 C = bins // 2 - 1
    """
    C = bins // 2 - 1
    x = (xq / C * max1)
    return x

# ============================================================
# 批量反量化 — 所有层一次处理
# ============================================================
def do_dequantize(t: torch.Tensor, bins: torch.Tensor, maxtensors: torch.Tensor):
    """
    对解码后的 int8 值做反量化，恢复成 FP16

    参数:
        t:          解码后的 int8 值 [nlayers, ntokens, nchannels]
        bins:       每层量化精度 [nlayers]  如 [32,32,...,16,16]
        maxtensors: 每 token 的最大绝对值 [nlayers, ntokens, 1]

    返回:
        反量化后的 FP16 值 [nlayers, ntokens, nchannels]

    反量化公式（编码时量化公式的逆）:
        编码时: xq = round(input * (MAX/max1) + MAX)
        解码时: input ≈ (xq - MAX) / MAX * max1

        其中 MAX = bins // 2 - 1 (来自 per-layer bins)
    """
    C = (bins // 2 - 1)[:, None, None]  # [nlayers, 1, 1]
    t = t - C                             # 减去偏移（恢复有符号值）
    t = t / C                             # 除以缩放因子
    t = t * maxtensors                    # 乘回原始最大值
    return t                              # 此时 t 约等于原始 FP16 值

# ============================================================
# 字节流 → GPU tensor（辅助）
# ============================================================
@_lmcache_nvtx_annotate
def bytes_to_tensor(bs: bytes, device="cuda") -> torch.Tensor:
    """numpy bytes → torch tensor on GPU"""
    np_array = np.frombuffer(bs, dtype=np.uint8)
    concated_string = torch.from_numpy(np_array).to(device)
    return concated_string

# ============================================================
# 重组字节（从变长压缩块恢复到固定大小 buffer）
# ============================================================
@_lmcache_nvtx_annotate
def recombine_bytes(bytes_tensor, output_lengths) -> torch.Tensor:
    """
    编码后每个 channel 长度不同（变长编码）。
    解码时需要先把这些变长字节填回固定大小的 buffer。

    输入:
        bytes_tensor:   所有有效字节拼接成的长串
        output_lengths: 每个 channel 的编码后长度 [nlayers, nchannels]

    返回:
        固定大小 buffer [nlayers, nchannels, BUFFER_SIZE]
    """
    output_buffer_size = CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK  # = 256
    offsets = output_lengths.flatten().cumsum(0).roll(1).reshape(output_lengths.shape)
    offsets[0][0] = 0
    indexes = torch.arange(output_buffer_size, device=offsets.device).tile(
        (output_lengths.shape[0], output_lengths.shape[1], 1))
    final_indexes = (indexes + offsets[:, :, None]).clamp(max = len(bytes_tensor) - 1)
    return bytes_tensor[final_indexes]

# ============================================================
# 解码一个 chunk（256 token）
# ============================================================
@_lmcache_nvtx_annotate
def decode_chunk(
        cdf: torch.Tensor,
        data_chunk: CacheGenGPUBytestream,
        target_buffer: torch.Tensor
    ) -> torch.Tensor:
    """
    用 CDF + 压缩字节 → GPU 算术解码 → int8 值写入 target_buffer

    输入:
        cdf:          概率表 [nlayers, nchannels, bins+1]
        data_chunk:   压缩数据 (bytestream + lengths + ntokens)
        target_buffer: 解码结果写到哪里 [nlayers, ntokens, nchannels]

    核心: torchac_cuda.decode_fast_prefsum() — CUDA kernel
    """
    bytes_tensor = data_chunk.bytestream
    # 计算每个 channel 的字节长度前缀和（定位各 channel 的数据）
    length_prefsum = data_chunk.bytestream_lengths.flatten().cumsum(0).reshape(
        data_chunk.bytestream_lengths.shape)

    # 调 CUDA kernel 做算术解码
    torchac_cuda.decode_fast_prefsum(
            cdf,              # CDF 概率表（编码时算的，和编码器共用）
            bytes_tensor,     # 压缩字节
            length_prefsum,   # 各 channel 的数据起止位置
            target_buffer)    # ★ 解码结果写到这里

# ============================================================
# decode_function_gpu — GPU 解码主函数
# ============================================================
@_lmcache_nvtx_annotate
def decode_function_gpu(
        cdf: torch.Tensor,
        data_chunks: List[CacheGenGPUBytestream],
        layers_in_key: int,          # = 32（Key 的层数）
        chunk_size: int,             # = 9500（总 token 数）
        output: torch.Tensor,        # 输出缓冲区 (ntokens, 2*nlayers*nchannels)
    ):
    """
    对所有 data_chunks 执行 GPU 算术解码，恢复出 Key 和 Value

    输入:
        cdf:           [nlayers, nchannels, bins+1]  概率表
        data_chunks:   编码时产出的压缩块列表（38个，每块256 token）
        layers_in_key: Key 的层数（=32）
        chunk_size:    总 token 数
        output:        预分配的输出缓冲区

    返回:
        key:   (32层, 9500词, 4096维)  — FP16
        value: (32层, 9500词, 4096维)  — FP16

    流程:
        1. 把 output 重塑为 (64, 9500, 4096)
        2. 逐个 data_chunk: decode_chunk → 填对应位置
        3. 重塑为 (2, 32, 9500, 4096) → 拆出 key 和 value
    """
    nlayers, nchannels, _ = cdf.shape   # nlayers=64, nchannels=4096

    # 准备空容器
    output = output.reshape((nlayers, chunk_size, nchannels))  # (64, 9500, 4096)

    # 逐个压缩块解码（每块 256 token）
    start = 0
    for data_chunk in data_chunks:
        end = start + data_chunk.ntokens   # 这块处理 [start, end) 区间的 token
        decode_chunk(cdf, data_chunk, output[:, start:end, :])
        start = end

    # 拆成 Key 和 Value
    out = output.reshape((2, layers_in_key, chunk_size, nchannels))  # (2, 32, 9500, 4096)
    key, value = out.float()  # uint8 → float

    return key, value   # 各 (32, 9500, 4096)

# ============================================================
# CacheGenDeserializer — 解码器对外的 API（run_cachegen.py 调用的就是这个）
# ============================================================
class CacheGenDeserializer(Deserializer):
    """
    CacheGen 反序列化器（解压器）

    使用方式:
        deserializer = CacheGenDeserializer(config, metadata)
        kv_tensor = deserializer.from_bytes(bytes)  # 压缩字节流 → 5D tensor
    """
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        # --- 跟编码器一样的配置（解压必须知道当时怎么压的） ---
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.output_buffer = None               # 输出缓冲区（惰性分配）
        self.fmt = metadata.fmt                 # = "huggingface"

        # --- 跟编码器一样的 per-layer bins 表 ---
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

    # ============================================================
    # 生成 Key 的 per-layer bins 数组（和编码器一模一样）
    # ============================================================
    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        """
        返回: [32,32,...,32, 16,16,...,16]
              └─ 前 first_layers 层 ─┘  └─ 其余 ─┘
        """
        ret = torch.zeros(config.key_third_layers)
        ret.fill_(config.key_third_bins)
        ret[:config.key_second_layers] = config.key_second_bins
        ret[:config.key_first_layers] = config.key_first_bins
        return ret.cuda()

    # ============================================================
    # 生成 Value 的 per-layer bins 数组（和编码器一模一样）
    # ============================================================
    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        """
        返回: [32,32, 16,16,16,...]
              └0-1┘ └── 其余 ──┘
        """
        ret = torch.zeros(config.key_third_layers)
        ret.fill_(config.value_second_bins)
        ret[:config.value_first_layers] = config.value_first_bins
        return ret.cuda()

    # ============================================================
    # 获取/分配输出缓冲区（惰性分配，只在第一次调用时创建）
    # ============================================================
    def get_output_buffer(self, nlayers: int, nchannels: int, ntokens: int):
        """
        预分配 GPU 上的输出缓冲区。
        只在大小变化时重新分配（不同数据可能 token 数不同）。
        """
        if self.output_buffer is None or self.output_buffer.shape[1] != 2 * nlayers * nchannels:
            self.output_buffer = torch.zeros(
                (self.chunk_size, 2 * nlayers * nchannels),
                dtype=torch.uint8
            ).cuda()
        return self.output_buffer[:ntokens, :]

    # ============================================================
    # from_bytes — 对外的解压接口
    # ============================================================
    @_lmcache_nvtx_annotate
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        """
        把压缩字节流恢复成 KV tensor

        输入:
            bs: 压缩字节流（编码器 to_bytes() 的返回值）

        返回:
            torch.Tensor: 5D tensor，格式取决于 self.fmt
              huggingface: (32, 2, 32, 9500, 128)
              vllm:        (32, 2, 9500, 32, 128)

        流程:
            1. pickle 反序列化 → 取出 cdf, data_chunks, max_tensors 等
            2. decode_function_gpu: GPU 算术解码 → int8 Key/Value
            3. do_dequantize: 反量化 → FP16 Key/Value
            4. stack + reshape + permute → 恢复 5D tensor
        """
        # --- 步骤1: pickle 反序列化 ---
        #     读回编码时保存的全部数据
        encoder_output = CacheGenGPUEncoderOutput.from_bytes(bs)
        encoder_output.max_tensors_key = encoder_output.max_tensors_key.cuda()
        encoder_output.max_tensors_value = encoder_output.max_tensors_value.cuda()

        # --- 步骤2: GPU 算术解码 ---
        ntokens = encoder_output.max_tensors_key.shape[1]           # = 9500
        layers_in_key = encoder_output.max_tensors_key.shape[0]     # = 32
        key, value = decode_function_gpu(
                encoder_output.cdf,              # 编码时的 CDF 表
                encoder_output.data_chunks,      # 压缩块列表
                layers_in_key,                   # = 32
                ntokens,                         # = 9500
                self.get_output_buffer(           # 输出缓冲区
                    encoder_output.cdf.shape[0] // 2,  # = 32
                    encoder_output.cdf.shape[1],       # = 4096
                    ntokens
                )
            )

        # --- 步骤3: 反量化（int8 → FP16） ---
        key = do_dequantize(key, self.key_bins, encoder_output.max_tensors_key)
        value = do_dequantize(value, self.value_bins, encoder_output.max_tensors_value)

        # --- 步骤4: 组装回 5D tensor ---
        #     stack: K 和 V → (2, 32, 9500, 4096)
        #     reshape: 4096 → (32头, 128维) → (2, 32, 9500, 32, 128)
        #     permute: 还原到目标框架的维度顺序
        nlayers, ntokens, nchannels = key.shape
        rng = nvtx.start_range("stack KV")
        blob = torch.stack([key, value])   # [2, nlayers, ntokens, nchannels]
        nvtx.end_range(rng)
        blob = blob.reshape((2, nlayers, ntokens, encoder_output.num_heads, encoder_output.head_size))
        # [2, 32, 9500, 32, 128]

        match self.fmt:
            case "vllm":
                # vLLM 格式: (32, 2, 9500, 32, 128)
                return blob.permute((1, 0, 2, 3, 4)).to(torch.bfloat16)
            case "huggingface":
                # HuggingFace 格式: (32, 2, 32, 9500, 128)
                return blob.permute((1, 0, 3, 2, 4)).to(torch.float16)
