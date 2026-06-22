import abc
import time

import torch

from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate


logger = init_logger(__name__)


class Serializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_bytes(self, t: torch.Tensor) -> bytes:
        raise NotImplementedError


class SerializerDebugWrapper(Serializer):
    def __init__(self, serializer: Serializer):
        self.serializer = serializer

    def to_bytes(self, t: torch.Tensor) -> bytes:
        start = time.perf_counter()
        bs = self.serializer.to_bytes(t)
        end = time.perf_counter()
        logger.debug(f"Serialization took {end - start:.2f} seconds")
        return bs


class Deserializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        raise NotImplementedError


class DeserializerDebugWrapper(Deserializer):
    def __init__(self, deserializer: Deserializer):
        self.deserializer = deserializer

    @_lmcache_nvtx_annotate
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        start = time.perf_counter()
        ret = self.deserializer.from_bytes(bs)
        end = time.perf_counter()
        logger.debug(f"Deserialization took {(end - start) * 1000:.2f} ms")
        return ret
