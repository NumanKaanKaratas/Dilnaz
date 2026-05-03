from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .cache import NazBackboneLayerCache


class SemanticDeltaMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.linear_key_head_dim != config.linear_value_head_dim:
            raise ValueError("linear key/value head dims must match in SemanticDeltaMixer")
        if config.linear_num_key_heads != config.linear_num_value_heads:
            raise ValueError("linear key/value head counts must match in SemanticDeltaMixer")
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_key_heads
        self.head_dim = config.linear_key_head_dim
        self.inner_size = self.num_heads * self.head_dim
        if self.inner_size != config.hidden_size:
            raise ValueError("linear_num_key_heads * linear_key_head_dim must equal hidden_size")
        self.conv_kernel_size = config.linear_conv_kernel_size

        self.q_proj = nn.Linear(self.hidden_size, self.inner_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.inner_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.inner_size, bias=False)
        self.alpha_proj = nn.Linear(self.hidden_size, self.inner_size, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.inner_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.inner_size, bias=False)
        self.out_proj = nn.Linear(self.inner_size, self.hidden_size, bias=False)
        self.conv_weight = nn.Parameter(torch.zeros(self.inner_size * 3, 1, self.conv_kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(self.inner_size * 3))
        with torch.no_grad():
            self.conv_weight[:, 0, -1].fill_(1.0)

    def _causal_conv(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[NazBackboneLayerCache],
        use_cache: bool,
    ) -> torch.Tensor:
        if self.conv_kernel_size == 1:
            return hidden_states
        batch_size = hidden_states.shape[0]
        if use_cache and cache is not None and cache.conv_state is not None:
            prefix = cache.conv_state
        else:
            prefix = hidden_states.new_zeros(batch_size, self.conv_kernel_size - 1, hidden_states.shape[-1])
        combined = torch.cat((prefix, hidden_states), dim=1)
        conved = F.conv1d(
            combined.transpose(1, 2),
            self.conv_weight,
            self.conv_bias,
            groups=hidden_states.shape[-1],
        ).transpose(1, 2)
        if use_cache and cache is not None:
            cache.conv_state = combined[:, -(self.conv_kernel_size - 1) :]
        return conved

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[NazBackboneLayerCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        mixed = torch.cat(
            (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ),
            dim=-1,
        )
        mixed = self._causal_conv(mixed, cache, use_cache)
        query_states, key_states, value_states = torch.chunk(mixed, 3, dim=-1)
        query_states = F.normalize(self._shape(query_states), p=2.0, dim=-1)
        key_states = F.normalize(self._shape(key_states), p=2.0, dim=-1)
        value_states = self._shape(value_states)
        alpha = torch.sigmoid(self._shape(self.alpha_proj(hidden_states)))
        beta = torch.sigmoid(self._shape(self.beta_proj(hidden_states)))
        updates = beta * key_states * value_states

        if use_cache and cache is not None and cache.delta_state is not None:
            state_prefix = cache.delta_state.unsqueeze(1)
            states = state_prefix + updates.cumsum(dim=1)
        else:
            states = updates.cumsum(dim=1)
        if use_cache and cache is not None:
            cache.delta_state = states[:, -1]

        output = query_states * (alpha * states + (1.0 - alpha) * value_states)
        gate = torch.sigmoid(self._shape(self.gate_proj(hidden_states)))
        output = (output * gate).reshape(hidden_states.shape)
        return self.out_proj(output)
