from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...common.norms import DilRMSNorm


class DilPackedCausalDepthwiseConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, bias: bool):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(torch.empty(hidden_size, self.kernel_size))
        self.bias = nn.Parameter(torch.empty(hidden_size)) if bias else None

    def forward(self, hidden_states: torch.Tensor, unit_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, surface_width, hidden_size = hidden_states.shape
        device = hidden_states.device
        positions = torch.arange(surface_width, device=device)
        offsets = torch.arange(1 - self.kernel_size, 1, device=device)
        raw_neighbor_pos = positions.unsqueeze(0) + offsets.unsqueeze(1)
        neighbor_pos = raw_neighbor_pos.clamp(0, surface_width - 1)
        in_bounds = raw_neighbor_pos.ge(0) & raw_neighbor_pos.lt(surface_width)
        hidden_expanded = hidden_states[:, neighbor_pos]
        same_unit = unit_ids[:, neighbor_pos].eq(unit_ids.unsqueeze(1)) & in_bounds.unsqueeze(0) & mask[:, neighbor_pos] & mask.unsqueeze(1)
        weight_view = self.weight.T.view(1, self.kernel_size, 1, hidden_size)
        output = (hidden_expanded * same_unit.unsqueeze(-1).to(hidden_states.dtype) * weight_view).sum(dim=1)
        if self.bias is not None:
            output = output + self.bias.view(1, 1, hidden_size)
        return output * mask.unsqueeze(-1).to(output.dtype)

    def step(self, hidden_state: torch.Tensor, cache: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, hidden_size = hidden_state.shape
        history_width = self.kernel_size - 1
        if cache is None:
            cache = hidden_state.new_zeros((batch_size, 0, hidden_size))
        if cache.shape[1] < history_width:
            pad = hidden_state.new_zeros((batch_size, history_width - cache.shape[1], hidden_size))
            cache_window = torch.cat([pad, cache, hidden_state.unsqueeze(1)], dim=1)
        else:
            cache_window = torch.cat([cache[:, -history_width:], hidden_state.unsqueeze(1)], dim=1)
        output = (cache_window * self.weight.T.view(1, self.kernel_size, hidden_size)).sum(dim=1)
        if self.bias is not None:
            output = output + self.bias.view(1, hidden_size)
        new_cache = cache_window[:, -history_width:] if history_width > 0 else cache_window[:, :0]
        return output, new_cache


class DilCausalConvSwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int, eps: float, bias: bool, dropout: float = 0.0):
        super().__init__()
        self.norm = DilRMSNorm(hidden_size, eps=eps)
        self.depthwise = DilPackedCausalDepthwiseConv(hidden_size, kernel_size, bias)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        unit_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = self.depthwise(hidden_states, unit_ids, mask)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.dropout(self.down_proj(hidden_states))
        hidden_states = residual + hidden_states
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states

    def step(self, hidden_state: torch.Tensor, cache: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        hidden_state, cache = self.depthwise.step(hidden_state, cache)
        hidden_state = F.silu(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        hidden_state = self.dropout(self.down_proj(hidden_state))
        return residual + hidden_state, cache
