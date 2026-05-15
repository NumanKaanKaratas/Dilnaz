import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...common.norms import DilRMSNorm


class DilAdaLNModulation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size * 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if condition.dim() == hidden_states.dim() - 1:
            condition = condition.unsqueeze(-2)
        shift, scale, gate = self.proj(condition).chunk(3, dim=-1)
        return hidden_states * (1.0 + scale) + shift, gate


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
        neighbor_pos = (positions.unsqueeze(0) + offsets.unsqueeze(1)).clamp(0, surface_width - 1)
        in_bounds = (positions.unsqueeze(0) + offsets.unsqueeze(1)).ge(0) & (positions.unsqueeze(0) + offsets.unsqueeze(1)).lt(surface_width)
        hidden_expanded = hidden_states[:, neighbor_pos]
        same_unit = unit_ids[:, neighbor_pos].eq(unit_ids.unsqueeze(1)) & in_bounds.unsqueeze(0) & mask[:, neighbor_pos] & mask.unsqueeze(1)
        weight_view = self.weight.T.view(1, self.kernel_size, 1, hidden_size)
        output = (hidden_expanded * same_unit.unsqueeze(-1).to(hidden_states.dtype) * weight_view).sum(dim=1)
        if self.bias is not None:
            output = output + self.bias.view(1, 1, hidden_size)
        return output * mask.unsqueeze(-1).to(output.dtype)


class DilCausalAdaLNConvSwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int, eps: float, bias: bool, dropout: float = 0.0):
        super().__init__()
        self.norm = DilRMSNorm(hidden_size, eps=eps)
        self.adaln = DilAdaLNModulation(hidden_size)
        self.depthwise = DilPackedCausalDepthwiseConv(hidden_size, kernel_size, bias)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        unit_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, residual_gate = self.adaln(self.norm(hidden_states), condition)
        if mask is None:
            mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = self.depthwise(hidden_states, unit_ids, mask)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = residual + self.dropout(hidden_states) * residual_gate
        return hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)


class DilWriterWordMixerBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, eps: float, bias: bool, dropout: float):
        super().__init__()
        self.attn_norm = DilRMSNorm(hidden_size, eps=eps)
        self.attn_adaln = DilAdaLNModulation(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = DilRMSNorm(hidden_size, eps=eps)
        self.ffn_adaln = DilAdaLNModulation(hidden_size)
        intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.ffn_dropout = nn.Dropout(dropout)

    def _math_attention(self, hidden_states: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        qkv = F.linear(hidden_states, self.attn.in_proj_weight, self.attn.in_proj_bias)
        query, key, value = qkv.chunk(3, dim=-1)
        head_count = self.attn.num_heads
        head_dim = hidden_size // head_count
        query = query.view(batch_size, seq_len, head_count, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, head_count, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, head_count, head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, p=self.attn.dropout, training=self.training)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        return F.linear(output, self.attn.out_proj.weight, self.attn.out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if window_mask is not None:
            safe_mask = window_mask.bool().clone()
            empty_rows = ~safe_mask.any(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
            hidden_states = hidden_states * safe_mask.unsqueeze(-1).to(hidden_states.dtype)
            key_padding_mask = ~safe_mask
        else:
            safe_mask = None
            key_padding_mask = None

        residual = hidden_states
        attn_input, attn_gate = self.attn_adaln(self.attn_norm(hidden_states), condition)
        attn_output = self._math_attention(attn_input, key_padding_mask)
        hidden_states = residual + self.attn_dropout(attn_output) * attn_gate

        residual = hidden_states
        hidden_states, ffn_gate = self.ffn_adaln(self.ffn_norm(hidden_states), condition)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = residual + self.ffn_dropout(self.down_proj(hidden_states)) * ffn_gate
        if safe_mask is not None:
            hidden_states = hidden_states * safe_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states

