from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .cache import NazBackboneLayerCache
from .normalization import ZeroCenteredRMSNorm
from .rotary import PartialRotaryEmbedding


class SemanticGlobalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary = PartialRotaryEmbedding(
            self.head_dim,
            config.partial_rotary_factor,
            config.rope_theta,
        )

    def _shape_q(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim)

    def _shape_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        cache: Optional[NazBackboneLayerCache] = None,
        use_cache: bool = False,
        cache_position: int = 0,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        query_states = self.q_norm(self._shape_q(self.q_proj(hidden_states)))
        key_states = self.k_norm(self._shape_kv(self.k_proj(hidden_states)))
        value_states = self._shape_kv(self.v_proj(hidden_states))
        query_states, key_states = self.rotary(query_states, key_states, position_ids)

        if use_cache and cache is not None:
            cache_start = cache_position
            cache_end = cache_start + query_length
            cache.ensure_kv_capacity(
                batch_size,
                cache_end,
                self.num_key_value_heads,
                self.head_dim,
                key_states.device,
                key_states.dtype,
            )
            cache.key[:, cache_start:cache_end].copy_(key_states)
            cache.value[:, cache_start:cache_end].copy_(value_states)
            key_states = cache.key[:, :cache_end]
            value_states = cache.value[:, :cache_end]

        key_length = key_states.shape[1]
        key_positions = torch.arange(key_length, device=hidden_states.device)
        attention_mask_4d = key_positions.view(1, 1, 1, key_length) <= position_ids.view(
            batch_size,
            1,
            query_length,
            1,
        )
        if attention_mask is not None and attention_mask.shape[1] == key_length:
            attention_mask_4d = attention_mask_4d & attention_mask[:, None, None, :]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask_4d,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
            enable_gqa=self.num_key_value_groups > 1,
        ).transpose(1, 2)
        gate = torch.sigmoid(self._shape_q(self.gate_proj(hidden_states)))
        output = (output * gate).reshape(batch_size, query_length, self.num_heads * self.head_dim)
        return self.o_proj(output)
