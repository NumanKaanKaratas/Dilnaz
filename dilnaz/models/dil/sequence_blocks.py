from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..common.norms import DilRMSNorm


class DilSequenceGatedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DilSequenceRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, partial_factor: float, theta: float):
        super().__init__()
        rotary_dim = int(head_dim * partial_factor)
        rotary_dim = max(2, rotary_dim - rotary_dim % 2)
        if rotary_dim > head_dim:
            raise ValueError("rotary dimension cannot exceed head_dim")
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim == 0:
            return query, key
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq)
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1).unsqueeze(1)
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1).unsqueeze(1)
        query_rot, query_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        key_rot, key_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]
        query_rot = (query_rot * cos) + (self._rotate_half(query_rot) * sin)
        key_rot = (key_rot * cos) + (self._rotate_half(key_rot) * sin)
        return torch.cat((query_rot, query_pass), dim=-1), torch.cat((key_rot, key_pass), dim=-1)

    @staticmethod
    def _rotate_half(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)


class DilBidirectionalAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        dropout: float,
        rms_norm_eps: float,
        partial_rotary_factor: float,
        rope_theta: float,
        attention_type: str,
        window_size: int,
    ):
        super().__init__()
        if attention_type not in {"global", "sliding"}:
            raise ValueError("DIL attention_type must be global or sliding")
        if num_heads % num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.scale = head_dim**-0.5
        self.attention_type = attention_type
        self.window_size = int(window_size)

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = DilRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = DilRMSNorm(head_dim, eps=rms_norm_eps)
        self.rotary = DilSequenceRotaryEmbedding(head_dim, partial_rotary_factor, rope_theta)

    def _shape_q(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = value.shape
        return value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = value.shape
        return value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    def _attention_mask(self, unit_mask: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        key_mask = unit_mask[:, None, None, :]
        if self.attention_type == "global" or seq_len <= self.window_size:
            return key_mask
        positions = torch.arange(seq_len, device=device)
        radius = max(self.window_size // 2, 1)
        local = (positions.view(1, -1) - positions.view(-1, 1)).abs().le(radius)
        return key_mask & local.view(1, 1, seq_len, seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        unit_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_norm(self._shape_q(self.q_proj(hidden_states)))
        key = self.k_norm(self._shape_kv(self.k_proj(hidden_states)))
        value = self._shape_kv(self.v_proj(hidden_states))
        query, key = self.rotary(query, key, position_ids)
        attention_mask = self._attention_mask(unit_mask, seq_len, hidden_states.device)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
            enable_gqa=self.num_key_value_groups > 1,
        )
        gate = torch.sigmoid(self._shape_q(self.gate_proj(hidden_states)))
        output = (output * gate).transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(output) * unit_mask.unsqueeze(-1).to(output.dtype)


class DilSequenceBlock(nn.Module):
    def __init__(self, config, attention_type: str):
        super().__init__()
        self.input_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = DilRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = DilBidirectionalAttention(
            hidden_size=config.hidden_size,
            num_heads=config.encoder_attention_heads,
            num_key_value_heads=config.encoder_key_value_heads,
            head_dim=config.encoder_head_dim,
            dropout=config.encoder_attention_dropout,
            rms_norm_eps=config.rms_norm_eps,
            partial_rotary_factor=config.encoder_partial_rotary_factor,
            rope_theta=config.encoder_rope_theta,
            attention_type=attention_type,
            window_size=config.encoder_attention_window,
        )
        self.feedforward = DilSequenceGatedMLP(config.hidden_size, config.encoder_intermediate_size, config.mlp_bias)
        self.dropout = nn.Dropout(config.dil_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        unit_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(
            self.attention(self.input_norm(hidden_states), unit_mask, position_ids)
        )
        hidden_states = hidden_states + self.dropout(self.feedforward(self.post_attention_norm(hidden_states)))
        return hidden_states * unit_mask.unsqueeze(-1).to(hidden_states.dtype)


class DilUnitContextBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        pattern = tuple(config.encoder_layer_pattern)
        if len(pattern) != config.encoder_context_layers:
            raise ValueError("encoder_layer_pattern length must equal encoder_context_layers")
        self.layers = nn.ModuleList([DilSequenceBlock(config, attention_type=item) for item in pattern])
        self.gradient_checkpointing = bool(config.encoder_gradient_checkpointing)

    def forward(
        self,
        hidden_states: torch.Tensor,
        unit_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
        layer_vectors = [] if output_hidden_states else None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and hidden_states.requires_grad:
                hidden_states = checkpoint(layer, hidden_states, unit_mask, position_ids, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states, unit_mask, position_ids)
            if layer_vectors is not None:
                layer_vectors.append(hidden_states)
        return hidden_states, tuple(layer_vectors) if layer_vectors is not None else None
