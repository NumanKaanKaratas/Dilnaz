from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NazBackboneLayerCache:
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    delta_state: Optional[torch.Tensor] = None
    conv_state: Optional[torch.Tensor] = None
    max_cache_length: Optional[int] = None

    def allocate_kv(
        self,
        batch_size: int,
        max_cache_length: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        shape = (batch_size, max_cache_length, num_key_value_heads, head_dim)
        self.key = torch.empty(shape, device=device, dtype=dtype)
        self.value = torch.empty(shape, device=device, dtype=dtype)

    def ensure_kv_capacity(
        self,
        batch_size: int,
        min_cache_length: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if self.max_cache_length is not None and min_cache_length > self.max_cache_length:
            raise ValueError(
                f"cache length {min_cache_length} exceeds max_cache_length {self.max_cache_length}"
            )
        if self.key is None or self.value is None:
            capacity = self.max_cache_length or min_cache_length
            self.allocate_kv(batch_size, capacity, num_key_value_heads, head_dim, device, dtype)
            return
        if self.key.shape[0] != batch_size:
            raise ValueError(f"cache batch size {self.key.shape[0]} != current batch size {batch_size}")
        if self.key.device != device or self.key.dtype != dtype:
            raise ValueError("cache device/dtype does not match current attention states")
        if min_cache_length <= self.key.shape[1]:
            return
        capacity = max(min_cache_length, self.key.shape[1] * 2)
        old_key = self.key
        old_value = self.value
        self.allocate_kv(batch_size, capacity, num_key_value_heads, head_dim, device, dtype)
        self.key[:, : old_key.shape[1]].copy_(old_key)
        self.value[:, : old_value.shape[1]].copy_(old_value)


@dataclass
class NazBackboneCache:
    layers: list[NazBackboneLayerCache]
    position: int = 0

    @classmethod
    def empty(
        cls,
        num_layers: int,
        *,
        batch_size: Optional[int] = None,
        max_cache_length: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        global_layer_indices: tuple[int, ...] = (),
    ) -> "NazBackboneCache":
        layers = [NazBackboneLayerCache(max_cache_length=max_cache_length) for _ in range(num_layers)]
        if max_cache_length is not None and batch_size is not None:
            if num_key_value_heads is None or head_dim is None or device is None or dtype is None:
                raise ValueError("preallocated KV cache requires num_key_value_heads, head_dim, device, and dtype")
            for layer_idx in global_layer_indices:
                layers[layer_idx].allocate_kv(
                    batch_size,
                    max_cache_length,
                    num_key_value_heads,
                    head_dim,
                    device,
                    dtype,
                )
        return cls(layers)
