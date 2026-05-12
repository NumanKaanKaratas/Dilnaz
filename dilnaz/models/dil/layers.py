from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..common.norms import DilRMSNorm


class DilGatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DilLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = DilGatedMLP(config)
        self.layernorm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DilConvSwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int, eps: float, bias: bool, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.norm = DilRMSNorm(hidden_size, eps=eps)
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_size,
            bias=bias,
        )
        self.gate_proj = nn.Conv1d(hidden_size, intermediate_size, kernel_size=1, bias=bias)
        self.up_proj = nn.Conv1d(hidden_size, intermediate_size, kernel_size=1, bias=bias)
        self.down_proj = nn.Conv1d(intermediate_size, hidden_size, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.depthwise(hidden_states)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states).transpose(1, 2)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class DilPackedDepthwiseConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, bias: bool):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.radius = self.kernel_size // 2
        self.weight = nn.Parameter(torch.empty(hidden_size, self.kernel_size))
        self.bias = nn.Parameter(torch.empty(hidden_size)) if bias else None

    def forward(self, hidden_states: torch.Tensor, unit_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, surface_width, hidden_size = hidden_states.shape
        positions = torch.arange(surface_width, device=hidden_states.device)
        output = hidden_states.new_zeros(hidden_states.shape)
        for kernel_idx, offset in enumerate(range(-self.radius, self.radius + 1)):
            source_positions = (positions + offset).clamp(0, surface_width - 1)
            in_bounds = (positions + offset).ge(0) & (positions + offset).lt(surface_width)
            source = hidden_states.index_select(1, source_positions)
            same_unit = unit_ids.eq(unit_ids.index_select(1, source_positions)) & in_bounds.view(1, -1) & mask
            output = output + source * same_unit.unsqueeze(-1).to(hidden_states.dtype) * self.weight[:, kernel_idx].view(1, 1, hidden_size)
        if self.bias is not None:
            output = output + self.bias.view(1, 1, hidden_size)
        return output * mask.unsqueeze(-1).to(output.dtype)


class DilPackedConvSwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int, eps: float, bias: bool, dropout: float = 0.0):
        super().__init__()
        self.norm = DilRMSNorm(hidden_size, eps=eps)
        self.depthwise = DilPackedDepthwiseConv(hidden_size, kernel_size, bias)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, unit_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = self.depthwise(hidden_states, unit_ids, mask)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.dropout(self.down_proj(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
