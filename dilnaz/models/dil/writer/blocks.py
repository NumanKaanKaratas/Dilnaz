from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...common.norms import DilRMSNorm
from ..layers import DilPackedDepthwiseConv


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
        self.norm = DilRMSNorm(hidden_size, eps=eps)
        self.adaln = DilAdaLNModulation(hidden_size)
        self.depthwise = DilPackedDepthwiseConv(hidden_size, kernel_size, bias)
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
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        else:
            mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        hidden_states = self.depthwise(hidden_states, unit_ids, mask)
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = residual + self.dropout(hidden_states) * residual_gate
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class DilByteStateCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, eps: float, dropout: float):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.query_norm = DilRMSNorm(hidden_size, eps=eps)
        self.state_norm = DilRMSNorm(hidden_size, eps=eps)
        self.adaln = DilAdaLNModulation(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        state_hidden: torch.Tensor,
        state_mask: torch.Tensor,
        query_unit_ids: torch.Tensor,
        state_unit_ids: torch.Tensor,
        byte_mask: torch.Tensor,
    ) -> torch.Tensor:
        query, residual_gate = self.adaln(self.query_norm(hidden_states), condition)
        batch_size, query_width, hidden_size = query.shape
        state_width = state_hidden.shape[1]
        query = self.q_proj(query).reshape(batch_size, query_width, self.heads, self.head_dim)
        keys = self.k_proj(self.state_norm(state_hidden)).reshape(batch_size, state_width, self.heads, self.head_dim)
        values = self.v_proj(state_hidden).reshape(batch_size, state_width, self.heads, self.head_dim)
        scores = torch.einsum("bqhd,bshd->bhqs", query, keys) / (self.head_dim**0.5)
        same_unit = query_unit_ids.unsqueeze(1).unsqueeze(-1).eq(state_unit_ids.unsqueeze(1).unsqueeze(2))
        valid = same_unit & state_mask.unsqueeze(1).unsqueeze(2) & byte_mask.unsqueeze(1).unsqueeze(-1)
        has_state = valid.any(dim=-1, keepdim=True)
        safe_valid = torch.where(has_state, valid, torch.ones_like(valid))
        scores = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype).masked_fill(~safe_valid, 0.0)
        attention = attention * has_state.to(attention.dtype)
        attn_output = torch.einsum("bhqs,bshd->bqhd", attention, values).reshape(batch_size, query_width, hidden_size)
        attn_output = self.out_proj(attn_output) * byte_mask.unsqueeze(-1).to(attn_output.dtype)
        hidden_states = hidden_states + self.dropout(attn_output) * residual_gate
        return hidden_states * byte_mask.unsqueeze(-1).to(hidden_states.dtype)


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

