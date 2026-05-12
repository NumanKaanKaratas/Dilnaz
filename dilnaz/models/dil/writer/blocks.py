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


class DilAdaLNConvSwiGLUBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, kernel_size: int, eps: float, bias: bool, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.norm = DilRMSNorm(hidden_size, eps=eps)
        self.adaln = DilAdaLNModulation(hidden_size)
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

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states, residual_gate = self.adaln(self.norm(hidden_states), condition)
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.depthwise(hidden_states)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states).transpose(1, 2)
        hidden_states = residual + self.dropout(hidden_states) * residual_gate
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class DilByteStateCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, eps: float, dropout: float):
        super().__init__()
        self.query_norm = DilRMSNorm(hidden_size, eps=eps)
        self.state_norm = DilRMSNorm(hidden_size, eps=eps)
        self.adaln = DilAdaLNModulation(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        state_hidden: torch.Tensor,
        state_mask: torch.Tensor,
        byte_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        safe_state_mask = state_mask.bool().clone()
        has_state = safe_state_mask.any(dim=1)
        empty_rows = ~has_state
        if empty_rows.any():
            safe_state_mask[empty_rows, 0] = True
        query, residual_gate = self.adaln(self.query_norm(hidden_states), condition)
        key_value = self.state_norm(state_hidden) * safe_state_mask.unsqueeze(-1).to(state_hidden.dtype)
        attn_output, _ = self.attn(
            query,
            key_value,
            key_value,
            key_padding_mask=~safe_state_mask,
            need_weights=False,
        )
        attn_output = attn_output * has_state.view(-1, 1, 1).to(attn_output.dtype)
        if byte_mask is not None:
            attn_output = attn_output * byte_mask.unsqueeze(-1).to(attn_output.dtype)
        hidden_states = hidden_states + self.dropout(attn_output) * residual_gate
        if byte_mask is not None:
            hidden_states = hidden_states * byte_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


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
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = residual + self.attn_dropout(attn_output) * attn_gate

        residual = hidden_states
        hidden_states, ffn_gate = self.ffn_adaln(self.ffn_norm(hidden_states), condition)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = residual + self.ffn_dropout(self.down_proj(hidden_states)) * ffn_gate
        if safe_mask is not None:
            hidden_states = hidden_states * safe_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states

