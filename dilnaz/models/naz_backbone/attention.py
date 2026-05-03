from __future__ import annotations

import math
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
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        query_states = self.q_norm(self._shape_q(self.q_proj(hidden_states)))
        key_states = self.k_norm(self._shape_kv(self.k_proj(hidden_states)))
        value_states = self._shape_kv(self.v_proj(hidden_states))
        query_states, key_states = self.rotary(query_states, key_states, position_ids)

        if use_cache and cache is not None:
            if cache.key is not None:
                key_states = torch.cat((cache.key, key_states), dim=1)
                value_states = torch.cat((cache.value, value_states), dim=1)
            cache.key = key_states
            cache.value = value_states

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)
        scores = torch.einsum("bthd,bshd->bhts", query_states, key_states) * self.scale

        key_length = key_states.shape[1]
        key_positions = torch.arange(key_length, device=hidden_states.device)
        causal_mask = key_positions.view(1, 1, 1, key_length) <= position_ids.view(
            batch_size,
            1,
            query_length,
            1,
        )
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        if attention_mask is not None and attention_mask.shape[1] == key_length:
            scores = scores.masked_fill(~attention_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores.float(), dim=-1).to(hidden_states.dtype)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        output = torch.einsum("bhts,bshd->bthd", attn, value_states)
        gate = torch.sigmoid(self._shape_q(self.gate_proj(hidden_states)))
        output = (output * gate).reshape(batch_size, query_length, self.num_heads * self.head_dim)
        return self.o_proj(output)
