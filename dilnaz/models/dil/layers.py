import torch
import torch.nn.functional as F
from torch import nn

from ..common.norms import DilRMSNorm


class DilPackedDepthwiseConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, bias: bool):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.radius = self.kernel_size // 2
        self.weight = nn.Parameter(torch.empty(hidden_size, self.kernel_size))
        self.bias = nn.Parameter(torch.empty(hidden_size)) if bias else None

    def forward(self, hidden_states: torch.Tensor, unit_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, surface_width, hidden_size = hidden_states.shape
        device = hidden_states.device
        positions = torch.arange(surface_width, device=device)
        offsets = torch.arange(-self.radius, self.radius + 1, device=device)
        neighbor_pos = (positions.unsqueeze(0) + offsets.unsqueeze(1)).clamp(0, surface_width - 1)
        in_bounds = (positions.unsqueeze(0) + offsets.unsqueeze(1)).ge(0) & (positions.unsqueeze(0) + offsets.unsqueeze(1)).lt(surface_width)
        hidden_expanded = hidden_states[:, neighbor_pos]
        same_unit = unit_ids[:, neighbor_pos].eq(unit_ids.unsqueeze(1)) & in_bounds.unsqueeze(0) & mask[:, neighbor_pos] & mask.unsqueeze(1)
        weight_view = self.weight.T.view(1, self.kernel_size, 1, hidden_size)
        output = (hidden_expanded * same_unit.unsqueeze(-1).to(hidden_states.dtype) * weight_view).sum(dim=1)
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


class DilLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(hidden_states)
